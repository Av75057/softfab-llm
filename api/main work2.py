import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

# === Конфиг ===

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000/v1")
API_KEY = os.getenv("API_KEY")

LOG_DIR = Path("/logs")

print(f"[INIT] VLLM_URL={VLLM_URL!r}")
print(f"[INIT] API_KEY set: {bool(API_KEY)}")
print(f"[INIT] LOG_DIR={LOG_DIR}")


# === Логирование ===

def log_interaction(payload: Dict[str, Any],
                    backend_response: Dict[str, Any],
                    status_code: int) -> None:
    """Пишем одну строку JSON в /logs/llm-YYYY-MM-DD.jsonl."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.utcnow()
        log_path = LOG_DIR / f"llm-{now.date().isoformat()}.jsonl"

        record = {
            "id": str(uuid.uuid4()),
            "ts_utc": now.isoformat() + "Z",
            "status_code": status_code,
            "request": payload,
            "response": backend_response,
        }

        line = json.dumps(record, ensure_ascii=False)
        print(f"[LOG] Writing record to {log_path}: {line[:80]}...")

        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception as e:
        print(f"[LOG] Failed to write log record: {e!r}")


# === FastAPI ===

app = FastAPI(
    title="SoftFab LLM API",
    description="Простой прокси к vLLM /v1 с логированием и стримингом",
    version="0.3.0",
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/backend/health")
async def backend_health():
    """Проверка доступности vLLM из API-контейнера."""
    url = VLLM_URL.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        return {
            "ok": resp.status_code == 200,
            "status_code": resp.status_code,
            "text": resp.text[:500],
            "url": url,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}


@app.get("/log-test")
async def log_test():
    """Тест: проверяем, что log_interaction пишет в /logs."""
    dummy_req = {"test": "ping"}
    dummy_resp = {"message": "pong"}
    log_interaction(dummy_req, dummy_resp, 200)
    return {"ok": True}


async def _chat_proxy_impl(request: Request):
    """Общий обработчик для /v1/chat/completions и /chat/completions."""

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    url = VLLM_URL.rstrip("/") + "/chat/completions"
    stream = bool(body.get("stream"))
    print(f"[CHAT] Proxy to {url}, stream={stream}")

    # --- STREAM ---
    if stream:
        async def event_generator():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            # просто прокидываем SSE-байты как есть
                            yield chunk
            except httpx.RequestError as e:
                print(f"[CHAT] Streaming error contacting vLLM: {e!r}")
                return

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    # --- NON-STREAM ---
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body)
    except httpx.RequestError as e:
        print(f"[CHAT] Error contacting vLLM: {e!r}")
        raise HTTPException(status_code=502, detail=f"Error contacting vLLM backend: {e}")

    try:
        backend_json = resp.json()
    except Exception as e:
        print(f"[CHAT] Failed to parse backend JSON: {e!r}, body={resp.text[:500]}")
        raise HTTPException(status_code=502, detail="Backend returned non-JSON response")

    log_interaction(body, backend_json, resp.status_code)
    return JSONResponse(status_code=resp.status_code, content=backend_json)


# OpenAI-совместный путь (для base_url="http://.../v1")
@app.post("/v1/chat/completions")
async def chat_proxy_v1(request: Request):
    return await _chat_proxy_impl(request)


# Прямой путь (для curl на /chat/completions)
@app.post("/chat/completions")
async def chat_proxy(request: Request):
    return await _chat_proxy_impl(request)


# === Простой веб-чат ===

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    return """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <title>SoftFab LLM Chat</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      padding: 12px 16px;
      background: #020617;
      border-bottom: 1px solid #1f2937;
      font-size: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    header span.title {
      font-weight: 600;
    }
    header span.status {
      font-size: 12px;
      color: #9ca3af;
    }
    #chat-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px;
      overflow: hidden;
    }
    #messages {
      flex: 1;
      border-radius: 12px;
      background: #020617;
      border: 1px solid #1f2937;
      padding: 12px;
      overflow-y: auto;
      font-size: 14px;
    }
    .message {
      margin-bottom: 10px;
      padding: 8px 10px;
      border-radius: 8px;
      max-width: 80%;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .message.user {
      background: #1e293b;
      margin-left: auto;
    }
    .message.assistant {
      background: #111827;
      margin-right: auto;
    }
    .message.system {
      background: #0f172a;
      color: #9ca3af;
      margin: 0 auto 10px auto;
      font-size: 12px;
      max-width: 90%;
    }
    #input-area {
      margin-top: 12px;
      display: flex;
      gap: 8px;
      align-items: flex-end;
    }
    #prompt {
      flex: 1;
      background: #020617;
      color: #e5e7eb;
      border-radius: 10px;
      border: 1px solid #1f2937;
      padding: 8px 10px;
      font-size: 14px;
      resize: vertical;
      min-height: 40px;
      max-height: 160px;
    }
    #send-btn {
      background: #4f46e5;
      color: white;
      border: none;
      border-radius: 10px;
      padding: 10px 16px;
      font-size: 14px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    #send-btn:disabled {
      opacity: 0.5;
      cursor: default;
    }
    #send-btn span.icon {
      font-size: 16px;
    }
    #footer-info {
      margin-top: 4px;
      font-size: 11px;
      color: #6b7280;
    }
  </style>
</head>
<body>
  <header>
    <span class="title">SoftFab · Local Qwen Chat</span>
    <span class="status" id="status">Готов</span>
  </header>

  <div id="chat-container">
    <div id="messages">
      <div class="message system">
        Сессия использует локальный Qwen2.5 через vLLM. Сообщения остаются только на вашем сервере.
      </div>
    </div>

    <div id="input-area">
      <textarea id="prompt" placeholder="Напишите запрос..."></textarea>
      <button id="send-btn" onclick="sendMessage()">
        <span class="icon">▶</span>
        <span>Отправить</span>
      </button>
    </div>
    <div id="footer-info">
      Модель: Qwen2.5 · streaming · endpoint: <code>/v1/chat/completions</code>
    </div>
  </div>

  <script>
    const messagesEl = document.getElementById("messages");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const statusEl = document.getElementById("status");

    const MODEL = "Qwen/Qwen2.5-32B-Instruct";

    const conversation = [
  {
    role: "system",
    content:
      "Ты локальная LLM-модель Qwen2.5, запущенная на моём сервере через vLLM. " +
      "Отвечай СТРОГО на русском языке. " +
      "Никогда не используй китайский язык (иероглифы) ни в каких частях ответа. " +
      "Если запрос приходит на другом языке — сначала мысленно переведи его, а отвечай по-русски. " +
      "Код и идентификаторы можешь писать на английском, но комментарии и пояснения — только на русском."
  }
];

    function appendMessage(role, text) {
      const div = document.createElement("div");
      div.classList.add("message", role);
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }

    function stripChinese(text) {
      // удаляем все символы из базовых китайских диапазонов Unicode
      return text.replace(/[\u3400-\u4dbf\u4e00-\u9fff]+/g, "[удалён китайский текст]");
    }

    async function sendMessage() {
      const prompt = promptEl.value.trim();
      if (!prompt || sendBtn.disabled) return;

      const userDiv = appendMessage("user", prompt);
      conversation.push({ role: "user", content: prompt });

      const assistantDiv = appendMessage("assistant", "");
      promptEl.value = "";
      promptEl.focus();

      sendBtn.disabled = true;
      statusEl.textContent = "Генерация...";

      try {
        const body = {
          model: MODEL,
          stream: true,
          messages: conversation
        };

        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(body)
        });

        if (!response.ok || !response.body) {
          assistantDiv.textContent = "[Ошибка] " + response.status + " " + response.statusText;
          statusEl.textContent = "Ошибка";
          sendBtn.disabled = false;
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const parts = buffer.split("\\n\\n");
          buffer = parts.pop();

          for (const part of parts) {
            const line = part.trim();
            if (!line) continue;
            if (!line.startsWith("data:")) continue;

            const data = line.slice(5).trim();
            if (data === "[DONE]") {
              break;
            }

            try {
              const json = JSON.parse(data);
              const delta = json.choices?.[0]?.delta;
              if (delta && delta.content) {
               assistantDiv.textContent = stripChinese(
                 assistantDiv.textContent + delta.content
               );
               messagesEl.scrollTop = messagesEl.scrollHeight;
              }

            } catch (e) {
              console.error("Failed to parse SSE chunk:", e, data);
            }
          }
        }

        const fullReply = assistantDiv.textContent;
        conversation.push({ role: "assistant", content: fullReply });
        statusEl.textContent = "Готово";
      } catch (e) {
        console.error(e);
        assistantDiv.textContent = "[Ошибка соединения с backend]";
        statusEl.textContent = "Ошибка сети";
      } finally {
        sendBtn.disabled = false;
      }
    }

    promptEl.addEventListener("keyup", (e) => {
      if (e.key !== "Enter") return;
      if (!e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>
    """

