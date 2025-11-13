import os
import json
import uuid
import datetime
import asyncio
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

# === Глобальное состояние здоровья backend-а ===

last_ok_time: datetime.datetime | None = None
last_status: str = "unknown"


def render_status_html() -> str:
    """Простая HTML-страница со статусом LLM."""
    now = datetime.datetime.utcnow()
    if last_ok_time is None:
        last_ok_str = "нет данных"
    else:
        last_ok_str = last_ok_time.isoformat(timespec="seconds") + "Z"

    color = "#4caf50" if last_status == "ok" else "#ff5050"
    msg = (
        "Модель Qwen2.5-32B доступна"
        if last_status == "ok"
        else "Модель Qwen2.5-32B временно недоступна"
    )

    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
      <meta charset="UTF-8" />
      <title>Статус LLM</title>
      <meta http-equiv="refresh" content="10" />
      <style>
        body {{
          font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
          background: #0f172a;
          color: #e5e7eb;
          margin: 0;
          padding: 0;
          display: flex;
          flex-direction: column;
          height: 100vh;
          justify-content: center;
          align-items: center;
          text-align: center;
        }}
        h1 {{
          color: {color};
          margin-bottom: 0.5rem;
        }}
        p {{
          color: #9ca3af;
          margin: 0.25rem 0;
        }}
        code {{
          color: #e5e7eb;
          background: #020617;
          padding: 2px 4px;
          border-radius: 4px;
        }}
      </style>
    </head>
    <body>
      <h1>{msg}</h1>
      <p>Последний успешный ответ от vLLM: <code>{last_ok_str}</code></p>
      <p>Текущее время (UTC): <code>{now.isoformat(timespec="seconds")}Z</code></p>
      <p>Страница обновляется автоматически каждые 10 секунд</p>
    </body>
    </html>
    """


async def ping_vllm() -> bool:
    """Пробный запрос к /models у vLLM."""
    url = VLLM_URL.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        return resp.status_code == 200
    except Exception as e:
        print(f"[HEALTH] Error pinging vLLM: {e!r}")
        return False


async def health_loop():
    """Фоновая корутина, которая каждые 10 сек обновляет статус."""
    global last_ok_time, last_status
    while True:
        ok = await ping_vllm()
        if ok:
            last_ok_time = datetime.datetime.utcnow()
            last_status = "ok"
        else:
            last_status = "down"
        await asyncio.sleep(10)


# === Логирование ===

def log_interaction(
    payload: Dict[str, Any],
    backend_response: Dict[str, Any],
    status_code: int,
) -> None:
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
    version="0.4.0",
)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(health_loop())


# Корневая страница: HTML-статус
@app.get("/", response_class=HTMLResponse)
async def root_page():
    return render_status_html()


# JSON-health для автоматических проверок
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": last_status,
        "last_ok_utc": last_ok_time.isoformat() + "Z" if last_ok_time else None,
    }


@app.get("/backend/health")
async def backend_health():
    """Проверка доступности vLLM из API-контейнера."""
    url = VLLM_URL.rstrip("/") + "/models"
    global last_ok_time, last_status
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        ok = resp.status_code == 200
        if ok:
            last_ok_time = datetime.datetime.utcnow()
            last_status = "ok"
        else:
            last_status = "down"
        return {
            "ok": ok,
            "status_code": resp.status_code,
            "text": resp.text[:500],
            "url": url,
        }
    except Exception as e:
        last_status = "down"
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

    # Если по состоянию видно, что backend лежит — сразу отдаём HTML-страницу
 #   if last_status == "down":
 #       print("[CHAT] vLLM status=down, returning HTML fallback")
 #       return HTMLResponse(render_status_html(), status_code=503)

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
            full_reply = []

            # На случай, если backend упал уже после начала стрима
            if last_status == "down":
                text = "data: " + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "Backend LLM сейчас недоступен. Откройте страницу /, чтобы посмотреть статус."
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
                yield (text + "\n\n").encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            # Для логов пытаемся собрать полный ответ
                            try:
                                text = chunk.decode("utf-8", errors="ignore")
                                for part in text.split("\n\n"):
                                    line = part.strip()
                                    if not line.startswith("data:"):
                                        continue
                                    data = line[5:].strip()
                                    if data == "[DONE]":
                                        continue
                                    try:
                                        j = json.loads(data)
                                        delta = j.get("choices", [{}])[0].get("delta", {})
                                        if isinstance(delta.get("content"), str):
                                            full_reply.append(delta["content"])
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            # Отдаём chunk наружу как есть
                            yield chunk

                # После окончания стрима логируем упрощённый ответ
                try:
                    backend_response = {
                        "stream": True,
                        "full_reply": "".join(full_reply),
                    }
                    log_interaction(body, backend_response, 200)
                except Exception as e:
                    print(f"[LOG] Failed to log stream reply: {e!r}")

            except httpx.RequestError as e:
                print(f"[CHAT] Streaming error contacting vLLM: {e!r}")
                err_text = "data: " + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "\n[Ошибка] Не удалось связаться с backend LLM."
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
                yield (err_text + "\n\n").encode("utf-8")
                yield b"data: [DONE]\n\n"
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
        # Вместо JSON-ошибки показываем HTML-страницу
        return HTMLResponse(render_status_html(), status_code=503)

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
    header span.status-ok {
      color: #4ade80;
    }
    header span.status-down {
      color: #f97373;
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
    .message.thinking {
      opacity: 0.8;
      font-style: italic;
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
      resize: none;
      min-height: 40px;
      max-height: 160px;
      line-height: 1.4;
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
      transition: background 0.15s ease, opacity 0.15s ease;
    }
    #send-btn:disabled {
      opacity: 0.5;
      cursor: default;
    }
    #send-btn:hover:not(:disabled) {
      background: #6366f1;
    }
    #footer-info {
      margin-top: 4px;
      font-size: 11px;
      color: #6b7280;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }
    #footer-info code {
      background: #020617;
      padding: 1px 4px;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <header>
    <span class="title">SoftFab · Local Qwen Chat</span>
    <span class="status" id="status">Проверка backend...</span>
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
        <span>Отправить</span>
      </button>
    </div>
    <div id="footer-info">
      <span>Модель: Qwen2.5 · streaming</span>
      <span>Enter — отправка, Shift+Enter — перенос строки</span>
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
          "Отвечай строго на русском языке. " +
          "Никогда не используй китайский язык (иероглифы) ни в каких частях ответа. " +
          "Если запрос приходит на другом языке — сначала мысленно переведи его, а отвечай по-русски. " +
          "Код и идентификаторы можешь писать на английском, но комментарии и пояснения — только на русском."
      }
    ];

    function appendMessage(role, text, extraClass) {
      const div = document.createElement("div");
      div.classList.add("message", role);
      if (extraClass) div.classList.add(extraClass);
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }

    function stripChinese(text) {
      return text.replace(/[\u3400-\u4dbf\u4e00-\u9fff]+/g, "[удален китайский текст]");
    }

    function setBackendStatus(ok, lastOkText) {
      statusEl.classList.remove("status-ok", "status-down");
      if (ok) {
        statusEl.classList.add("status-ok");
        statusEl.textContent = "Backend: доступен" + (lastOkText ? " (последний ответ: " + lastOkText + ")" : "");
      } else {
        statusEl.classList.add("status-down");
        statusEl.textContent = "Backend: недоступен";
      }
    }

    async function pollHealth() {
      try {
        const resp = await fetch("/health");
        if (!resp.ok) {
          setBackendStatus(false);
          return;
        }
        const data = await resp.json();
        const ok = data.status === "ok";
        const last = data.last_ok_utc || null;
        setBackendStatus(ok, last);
      } catch (e) {
        setBackendStatus(false);
      }
    }

    // авто-подстройка высоты textarea
    function autoResizeTextarea() {
      promptEl.style.height = "auto";
      const newHeight = Math.min(promptEl.scrollHeight, 160);
      promptEl.style.height = newHeight + "px";
    }

    promptEl.addEventListener("input", autoResizeTextarea);

    async function sendMessage() {
      const prompt = promptEl.value.trim();
      if (!prompt || sendBtn.disabled) return;

      appendMessage("user", prompt);
      conversation.push({ role: "user", content: prompt });

      const assistantDiv = appendMessage("assistant", "Модель думает...", "thinking");

      promptEl.value = "";
      autoResizeTextarea();
      promptEl.focus();

      sendBtn.disabled = true;
      statusEl.textContent = "Генерация (stream)...";

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

        const contentType = response.headers.get("content-type") || "";

        if (!response.ok) {
          const text = await response.text();
          assistantDiv.classList.remove("thinking");
          assistantDiv.textContent =
            "[Ошибка] " + response.status + " " + response.statusText + "\\n" +
            text.slice(0, 200);
          statusEl.textContent = "Ошибка";
          sendBtn.disabled = false;
          return;
        }

        if (!contentType.includes("text/event-stream")) {
          const text = await response.text();
          assistantDiv.classList.remove("thinking");
          assistantDiv.textContent =
            "[Ошибка] backend вернул не stream, а: " + text.slice(0, 200);
          statusEl.textContent = "Ошибка";
          sendBtn.disabled = false;
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let fullReply = "";

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
                fullReply += delta.content;
                assistantDiv.classList.remove("thinking");
                assistantDiv.textContent = stripChinese(fullReply);
                messagesEl.scrollTop = messagesEl.scrollHeight;
              }
            } catch (e) {
              console.error("Failed to parse SSE chunk:", e, data);
            }
          }
        }

        if (!fullReply) {
          fullReply = assistantDiv.textContent || "";
        }

        conversation.push({ role: "assistant", content: fullReply });
        statusEl.textContent = "Готово";
      } catch (e) {
        console.error(e);
        assistantDiv.classList.remove("thinking");
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

    // первый health-check и периодические обновления статуса
    pollHealth();
    setInterval(pollHealth, 10000);
  </script>
</body>
</html>
    """

