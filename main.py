import os
import json
import uuid
import datetime
import asyncio
import html
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# === Конфиг ===

# По умолчанию смотрим на локальный vLLM на 8001 (как ты запускаешь контейнер).
# В docker-compose можно переопределить: VLLM_URL=http://vllm:8000/v1
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1").rstrip("/")
API_KEY = os.getenv("API_KEY")  # если включишь ключи на vLLM
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
    url = VLLM_URL + "/models"
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_auth_headers()) as client:
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


# === Helpers ===

def _auth_headers() -> Dict[str, str]:
    """Заголовки авторизации к vLLM (если задан API_KEY)."""
    hdrs: Dict[str, str] = {}
    if API_KEY:
        hdrs["Authorization"] = f"Bearer {API_KEY}"
    return hdrs


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
    version="0.5.0",
)

# Немного CORS (если фронт вынесен отдельно)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    url = VLLM_URL + "/models"
    global last_ok_time, last_status
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_auth_headers()) as client:
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


# === /v1/models прокси (удобно для curl-проверок) ===
@app.get("/v1/models")
async def v1_models():
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=_auth_headers()) as client:
            r = await client.get(VLLM_URL + "/models")
        return ResponseWithHeaders(r)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e))


def ResponseWithHeaders(resp: httpx.Response) -> JSONResponse | HTMLResponse | StreamingResponse:
    """Унифицированный возврат ответа бэкенда с сохранением content-type."""
    media_type = resp.headers.get("content-type", "application/json")
    return JSONResponse(
        status_code=resp.status_code,
        content=resp.json() if "json" in media_type or media_type.startswith("application/json") else {"raw": resp.text},
        media_type=media_type,
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Простой HTML-дашборд: статус backend + последние запросы из логов за сегодня."""
    now = datetime.datetime.utcnow()
    status = last_status
    last_ok = last_ok_time.isoformat() + "Z" if last_ok_time else "нет данных"

    today = now.date()
    log_path = LOG_DIR / f"llm-{today.isoformat()}.jsonl"
    records: list[dict] = []

    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-100:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except Exception:
                    continue
        except Exception as e:
            print(f"[ADMIN] Failed to read logs: {e!r}")
    else:
        print(f"[ADMIN] Log file not found: {log_path}")

    total_calls = len(records)
    success_calls = sum(
        1 for r in records
        if isinstance(r.get("status_code"), int) and 200 <= r["status_code"] < 300
    )
    failed_calls = total_calls - success_calls

    table_rows = ""
    for rec in reversed(records[-20:]):
        ts = str(rec.get("ts_utc", ""))
        status_code = rec.get("status_code", "")

        req = rec.get("request", {}) or {}
        prompt_preview = ""
        messages = req.get("messages")
        if isinstance(messages, list):
            user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
            if user_msgs:
                prompt_preview = user_msgs[-1]
        if not prompt_preview:
            prompt_preview = str(req)
        prompt_preview = html.escape(str(prompt_preview)[:120])

        resp = rec.get("response", {}) or {}
        reply_preview = ""
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) or {}
            reply_preview = msg.get("content", "")

        if not reply_preview and resp.get("stream") and "full_reply" in resp:
            reply_preview = resp.get("full_reply", "")

        reply_preview = html.escape(str(reply_preview)[:120])

        table_rows += (
            f"<tr>"
            f"<td>{html.escape(ts)}</td>"
            f"<td>{status_code}</td>"
            f"<td>{prompt_preview}</td>"
            f"<td>{reply_preview}</td>"
            f"</tr>"
        )

    log_info = (
        f"Файл логов: {html.escape(str(log_path))} / записей: {total_calls}"
        if log_path.exists()
        else f"Файл логов за сегодня не найден: {html.escape(str(log_path))}"
    )

    html_page = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
      <meta charset="UTF-8" />
      <title>SoftFab LLM Admin</title>
      <style>
        body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; background: #020617; color: #e5e7eb; margin: 0; padding: 0; }}
        header {{ padding: 12px 16px; background: #020617; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; font-size: 14px; }}
        header .title {{ font-weight: 600; }}
        header .status-ok {{ color: #4ade80; }}
        header .status-down {{ color: #f97373; }}
        main {{ padding: 16px; }}
        .card {{ background: #020617; border: 1px solid #1f2937; border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; }}
        .card h2 {{ font-size: 16px; margin: 0 0 8px 0; }}
        .card p {{ margin: 4px 0; color: #9ca3af; font-size: 13px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th, td {{ padding: 6px 8px; border-bottom: 1px solid #1f2937; vertical-align: top; }}
        th {{ text-align: left; background: #020617; color: #9ca3af; }}
        tbody tr:nth-child(2n) {{ background: #020617; }}
        code {{ background: #020617; padding: 2px 4px; border-radius: 4px; }}
        .small {{ font-size: 11px; color: #6b7280; }}
        a {{ color: #93c5fd; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
      </style>
    </head>
    <body>
      <header>
        <div class="title">SoftFab · LLM Admin</div>
        <div><span>Текущее время (UTC): {html.escape(now.isoformat(timespec="seconds"))}Z</span></div>
      </header>
      <main>
        <div class="card">
          <h2>Статус backend</h2>
          <p>Состояние: <strong class="{ 'status-ok' if status == 'ok' else 'status-down' }">{html.escape(status)}</strong></p>
          <p>Последний успешный ответ от vLLM: <code>{html.escape(last_ok)}</code></p>
          <p class="small">См. <a href="/">/</a> и <a href="/backend/health">/backend/health</a>.</p>
        </div>
        <div class="card">
          <h2>Логи за сегодня</h2>
          <p>{log_info}</p>
          <p>Успешных запросов: {success_calls} / Ошибок: {failed_calls}</p>
        </div>
        <div class="card">
          <h2>Последние запросы (до 20)</h2>
          <div style="overflow-x: auto;">
            <table>
              <thead>
                <tr><th>Время (UTC)</th><th>HTTP</th><th>Пользовательский запрос</th><th>Ответ модели</th></tr>
              </thead>
              <tbody>{table_rows or '<tr><td colspan="4">Нет записей за сегодня.</td></tr>'}</tbody>
            </table>
          </div>
        </div>
      </main>
    </body>
    </html>
    """
    return HTMLResponse(html_page)


async def _chat_proxy_impl(request: Request):
    """Общий обработчик для /v1/chat/completions и /chat/completions."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    url = VLLM_URL + "/chat/completions"
    stream = bool(body.get("stream"))
    print(f"[CHAT] Proxy to {url}, stream={stream}")

    # --- STREAM ---
    if stream:
        async def event_generator():
            full_reply = []

            if last_status == "down":
                text = "data: " + json.dumps(
                    {"choices": [{"delta": {"content": "Backend LLM сейчас недоступен. Откройте / для статуса."}}]},
                    ensure_ascii=False,
                )
                yield (text + "\n\n").encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

            try:
                async with httpx.AsyncClient(timeout=None, headers=_auth_headers()) as client:
                    async with client.stream("POST", url, json=body) as resp:
                        # Важно: проверим, что это SSE
                        if resp.status_code >= 400:
                            err = await resp.aread()
                            yield f"data: {json.dumps({'error': resp.status_code, 'body': err.decode('utf-8','ignore')[:300]})}\n\n".encode("utf-8")
                            yield b"data: [DONE]\n\n"
                            return

                        async for chunk in resp.aiter_bytes():
                            # Собираем контент для логов
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

                            yield chunk

                try:
                    backend_response = {"stream": True, "full_reply": "".join(full_reply)}
                    log_interaction(body, backend_response, 200)
                except Exception as e:
                    print(f"[LOG] Failed to log stream reply: {e!r}")

            except httpx.RequestError as e:
                print(f"[CHAT] Streaming error contacting vLLM: {e!r}")
                err_text = "data: " + json.dumps({"choices": [{"delta": {"content": "\n[Ошибка] Не удалось связаться с backend LLM."}}]}, ensure_ascii=False)
                yield (err_text + "\n\n").encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- NON-STREAM ---
    try:
        async with httpx.AsyncClient(timeout=120.0, headers=_auth_headers()) as client:
            resp = await client.post(url, json=body)
    except httpx.RequestError as e:
        print(f"[CHAT] Error contacting vLLM: {e!r}")
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
      background: #0f172a; color: #e5e7eb; margin: 0; padding: 0;
      display: flex; flex-direction: column; height: 100vh;
    }
    header { padding: 12px 16px; background: #020617; border-bottom: 1px solid #1f2937;
      font-size: 14px; display: flex; align-items: center; justify-content: space-between; }
    header span.title { font-weight: 600; }
    header span.status { font-size: 12px; color: #9ca3af; }
    header span.status-ok { color: #4ade80; }
    header span.status-down { color: #f97373; }
    #chat-container { flex: 1; display: flex; flex-direction: column; padding: 16px; overflow: hidden; }
    #messages { flex: 1; border-radius: 12px; background: #020617; border: 1px solid #1f2937;
      padding: 12px; overflow-y: auto; font-size: 14px; }
    .message { margin-bottom: 10px; padding: 8px 10px; border-radius: 8px; max-width: 80%;
      white-space: pre-wrap; word-wrap: break-word; }
    .message.user { background: #1e293b; margin-left: auto; }
    .message.assistant { background: #111827; margin-right: auto; }
    .message.system { background: #0f172a; color: #9ca3af; margin: 0 auto 10px auto; font-size: 12px; max-width: 90%; }
    .message.thinking { opacity: 0.8; font-style: italic; }
    #input-area { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }
    #top-controls { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #9ca3af; margin-bottom: 4px; align-items: center; }
    #top-controls label { display: inline-flex; align-items: center; gap: 4px; }
    #role-select { background: #020617; border: 1px solid #1f2937; border-radius: 6px; padding: 2px 6px; color: #e5e7eb; font-size: 12px; }
    #clear-btn { background: #1f2937; color: #e5e7eb; border: 1px solid #374151; border-radius: 8px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
    #clear-btn:hover { background: #111827; border-color: #4b5563; }
    #clear-btn:disabled { opacity: 0.5; cursor: default; }
    #prompt-row { display: flex; gap: 8px; align-items: flex-end; }
    #prompt { flex: 1; background: #020617; color: #e5e7eb; border-radius: 10px; border: 1px solid #1f2937;
      padding: 8px 10px; font-size: 14px; resize: none; min-height: 40px; max-height: 160px; line-height: 1.4; }
    #send-btn { background: #4f46e5; color: white; border: none; border-radius: 10px; padding: 10px 16px; font-size: 14px; cursor: pointer; }
    #send-btn:disabled { opacity: 0.5; cursor: default; }
    #send-btn:hover:not(:disabled) { background: #6366f1; }
    #params-row { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #9ca3af; }
    #params-row input { background: #020617; border: 1px solid #1f2937; border-radius: 6px; padding: 2px 6px; color: #e5e7eb; width: 70px; font-size: 12px; }
    #footer-info { margin-top: 4px; font-size: 11px; color: #6b7280; display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
    #footer-info code { background: #020617; padding: 1px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <header>
    <span class="title">SoftFab · Local Qwen Chat</span>
    <span class="status" id="status">Проверка backend...</span>
  </header>

  <div id="chat-container">
    <div id="messages">
      <div class="message system" id="role-info">
        Роль: обычный ассистент. Сообщения обрабатываются локальной моделью Qwen2.5 через vLLM.
      </div>
    </div>

    <div id="input-area">
      <div id="top-controls">
        <label>
          Роль
          <select id="role-select">
            <option value="default">Обычный ассистент</option>
            <option value="coder">Помощник по коду</option>
            <option value="architect">Системный архитектор</option>
            <option value="math">Математик / теоретик</option>
            <option value="translator">Переводчик</option>
          </select>
        </label>
        <button id="clear-btn" type="button">Очистить диалог</button>
      </div>

      <div id="prompt-row">
        <textarea id="prompt" placeholder="Напишите запрос..."></textarea>
        <button id="send-btn" onclick="sendMessage()"><span>Отправить</span></button>
      </div>

      <div id="params-row">
        <label>Temperature <input type="number" id="temperature" min="0" max="2" step="0.1" value="0.7" /></label>
        <label>Max tokens <input type="number" id="max-tokens" min="1" max="4096" value="512" /></label>
      </div>
    </div>

    <div id="footer-info">
      <span>Модель: Qwen2.5-32B · streaming</span>
      <span>Enter — отправка, Shift+Enter — перенос строки</span>
    </div>
  </div>

  <script>
    const messagesEl = document.getElementById("messages");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send-btn");
    const clearBtn = document.getElementById("clear-btn");
    const statusEl = document.getElementById("status");
    const temperatureEl = document.getElementById("temperature");
    const maxTokensEl = document.getElementById("max-tokens");
    const roleSelectEl = document.getElementById("role-select");
    const roleInfoEl = document.getElementById("role-info");

    // ВАЖНО: это должно совпадать с --served-model-name у vLLM
    const MODEL = "qwen3next-80b-a3b-instruct";

    const ROLE_PRESETS = {
      default: {
        name: "Обычный ассистент",
        description: "Общие задачи, объяснения, помощь в текстах и обсуждениях.",
        system:
          "Ты локальная LLM-модель Qwen2.5, запущенная на моем сервере через vLLM. " +
          "Отвечай строго на русском языке. Никогда не используй китайский язык. " +
          "Если запрос пришел не на русском — отвечай по-русски. " +
          "Код и идентификаторы на английском допускаются; комментарии — на русском."
      },
      coder: {
        name: "Помощник по коду",
        description: "Генерация и разбор кода, поиск багов, рефакторинг.",
        system:
          "Ты опытный разработчик и ревьюер кода. Отвечай строго на русском языке. " +
          "Коротко идея, затем конкретный код с комментариями. " +
          "Если есть варианты — перечисли с плюсами/минусами."
      },
      architect: {
        name: "Системный архитектор",
        description: "Архитектура, инфраструктура, интеграции, high-level дизайн.",
        system:
          "Ты системный архитектор. Отвечай строго на русском языке. " +
          "Структурируй ответы: резюме → архитектура по слоям → риски/масштабирование → наблюдаемость/CI/CD/безопасность."
      },
      math: {
        name: "Математик / теоретик",
        description: "Решение и разбор математических задач.",
        system:
          "Ты математик-теоретик. Отвечай по-русски. " +
          "Переформулируй задачу, решай пошагово, при возможности дай альтернативные подходы."
      },
      translator: {
        name: "Переводчик",
        description: "Перевод текстов и адаптация стиля.",
        system:
          "Ты переводчик и редактор. Основной язык ответа — русский. " +
          "Переводи с сохранением смысла и стиля, делай формулировки естественными."
      }
    };

    let currentRoleKey = "default";
    const conversation = [{ role: "system", content: ROLE_PRESETS[currentRoleKey].system }];

    function updateRoleInfo() {
      const role = ROLE_PRESETS[currentRoleKey];
      roleInfoEl.textContent =
        "Роль: " + role.name + ". " + role.description + " Сообщения обрабатываются локальной моделью Qwen2.5 через vLLM.";
    }
    updateRoleInfo();

    roleSelectEl.addEventListener("change", () => {
      const newKey = roleSelectEl.value;
      if (!ROLE_PRESETS[newKey]) return;
      currentRoleKey = newKey;
      conversation[0].content = ROLE_PRESETS[currentRoleKey].system;
      updateRoleInfo();
      const div = document.createElement("div");
      div.classList.add("message", "system");
      div.textContent = "Роль изменена на: " + ROLE_PRESETS[currentRoleKey].name;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });

    function resetConversation() {
      conversation.length = 0;
      conversation.push({ role: "system", content: ROLE_PRESETS[currentRoleKey].system });

      const children = Array.from(messagesEl.children);
      for (const child of children) {
        if (child !== roleInfoEl) messagesEl.removeChild(child);
      }
      promptEl.value = "";
      autoResizeTextarea();
      statusEl.textContent = "Диалог очищен";
    }
    clearBtn.addEventListener("click", resetConversation);

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
        if (!resp.ok) { setBackendStatus(false); return; }
        const data = await resp.json();
        const ok = data.status === "ok";
        const last = data.last_ok_utc || null;
        setBackendStatus(ok, last);
      } catch (e) { setBackendStatus(false); }
    }

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
      clearBtn.disabled = true;
      statusEl.textContent = "Генерация...";

      let temperature = parseFloat(temperatureEl.value);
      if (Number.isNaN(temperature)) temperature = 0.7;
      let maxTokens = parseInt(maxTokensEl.value, 10);
      if (!Number.isFinite(maxTokens) || maxTokens <= 0) maxTokens = 512;

      try {
        const body = {
          model: MODEL,
          stream: false,
          temperature: temperature,
          max_tokens: maxTokens,
          messages: conversation
        };

        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });

        const contentType = response.headers.get("content-type") || "";

        if (!response.ok) {
          const text = await response.text();
          assistantDiv.classList.remove("thinking");
          assistantDiv.textContent = "[Ошибка] " + response.status + " " + response.statusText + "\\n" + text.slice(0, 200);
          statusEl.textContent = "Ошибка";
          return;
        }

        if (!contentType.includes("text/event-stream")) {
          const data = await response.json();
          const msg = (data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "";
          assistantDiv.classList.remove("thinking");
          assistantDiv.textContent = msg || "[Пустой ответ]";
          statusEl.textContent = "Готово";
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let fullReply = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: false });

          const parts = buffer.split("\\n\\n");
          buffer = parts.pop();

          for (const part of parts) {
            const line = part.trim();
            if (!line) continue;
            if (!line.startsWith("data:")) continue;

            const data = line.slice(5).trim();
            if (data === "[DONE]") { buffer = ""; break; }

            try {
              const json = JSON.parse(data);
              const delta = json.choices?.[0]?.delta;
              if (delta && delta.content) {
                fullReply += delta.content;
                assistantDiv.classList.remove("thinking");
                assistantDiv.textContent = stripChinese(fullReply);
                messagesEl.scrollTop = messagesEl.scrollHeight;
              }
            } catch (e) { console.error("Failed to parse SSE chunk:", e, data); }
          }
        }

        if (!fullReply) fullReply = assistantDiv.textContent || "";
        conversation.push({ role: "assistant", content: fullReply });
        statusEl.textContent = "Готово";
      } catch (e) {
        console.error(e);
        assistantDiv.classList.remove("thinking");
        assistantDiv.textContent = "[Ошибка соединения с backend]";
        statusEl.textContent = "Ошибка сети";
      } finally {
        sendBtn.disabled = false;
        clearBtn.disabled = false;
      }
    }

    promptEl.addEventListener("keyup", (e) => {
      if (e.key !== "Enter") return;
      if (!e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    pollHealth();
    setInterval(pollHealth, 10000);
  </script>
</body>
</html>
    """

