import asyncio
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
DB_PATH = os.getenv("POSTBOT_DB_PATH", "/data/postbot.db")
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY")  # e.g. http://user:pass@host:port
TELEGRAM_TIMEOUT = float(os.getenv("TELEGRAM_TIMEOUT", "30"))

LENGTH_OPTIONS: Dict[str, str] = {
    "short": "60-90 слов, одно-два абзаца",
    "medium": "120-180 слов, 2-4 абзаца",
    "long": "220-320 слов, 4-6 абзацев",
}


class ChatSettingsStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_settings (
                    chat_id INTEGER PRIMARY KEY,
                    style TEXT,
                    length TEXT DEFAULT 'medium',
                    last_topic TEXT
                )
                """
            )
            db.commit()

    async def get(self, chat_id: int) -> Dict[str, Optional[str]]:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT style, length, last_topic FROM chat_settings WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        if row:
            style, length, last_topic = row
            length = length or "medium"
        else:
            style, length, last_topic = None, "medium", None
        return {"style": style, "length": length, "last_topic": last_topic}

    async def set_style(self, chat_id: int, style: str) -> None:
        async with self._lock:
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    """
                    INSERT INTO chat_settings (chat_id, style)
                    VALUES (?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET style=excluded.style
                    """,
                    (chat_id, style),
                )
                db.commit()

    async def set_length(self, chat_id: int, length: str) -> None:
        async with self._lock:
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    """
                    INSERT INTO chat_settings (chat_id, length)
                    VALUES (?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET length=excluded.length
                    """,
                    (chat_id, length),
                )
                db.commit()

    async def set_last_topic(self, chat_id: int, topic: str) -> None:
        async with self._lock:
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    """
                    INSERT INTO chat_settings (chat_id, last_topic)
                    VALUES (?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET last_topic=excluded.last_topic
                    """,
                    (chat_id, topic),
                )
                db.commit()


def extract_args(message: types.Message) -> Optional[str]:
    if not message.text:
        return None
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[1].strip()


def build_system_prompt() -> str:
    return (
        "Ты пишешь ёмкие и увлекательные посты для Telegram-канала на русском языке. "
        "Избегай воды, используй ясный тон и структуру, добавляй микро-призывы к действию, "
        "но не используй эмодзи, если они не к месту."
    )


def build_user_prompt(topic: str, length: str, style: Optional[str]) -> str:
    length_hint = LENGTH_OPTIONS.get(length, LENGTH_OPTIONS["medium"])
    style_hint = style if style else "Используй нейтральный профессиональный стиль."
    return (
        f"Тема: {topic}\n"
        f"Длина: {length_hint}\n"
        f"Стиль: {style_hint}\n"
        "Сделай текст готовым к публикации."
    )


def split_chunks(text: str, limit: int = 3500) -> list[str]:
    # Split by lines first; if a single line is too long, hard-split it.
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

    for line in text.splitlines():
        if len(line) >= limit:
            flush()
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
            continue
        line_len = len(line) + 1  # account for newline
        if current_len + line_len > limit:
            flush()
        current.append(line)
        current_len += line_len

    flush()
    if not chunks:
        return [text]
    return chunks


def strip_thinking(text: str) -> str:
    """Drop model reasoning enclosed before </think> if present."""
    if "</think>" not in text:
        return text
    return text.split("</think>", 1)[1].lstrip()


async def generate_post(
    client: AsyncOpenAI, topic: str, length: str, style: Optional[str]
) -> str:
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(topic, length, style)},
        ],
        temperature=0.8,
    )
    content = response.choices[0].message.content or ""
    return strip_thinking(content.strip())


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is required")
    if not OPENAI_BASE_URL or not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required")

    store = ChatSettingsStore(DB_PATH)
    await store.init()

    bot = Bot(token=BOT_TOKEN, proxy=TELEGRAM_PROXY, parse_mode=None)
    dp = Dispatcher()
    client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    def log_message(message: types.Message) -> None:
        user = message.from_user
        user_repr = f"{user.id}" if user else "unknown"
        logger.info(
            "incoming message chat=%s user=%s text=%s",
            message.chat.id,
            user_repr,
            (message.text or "").strip(),
        )

    @dp.message(Command("start"))
    async def handle_start(message: types.Message) -> None:
        log_message(message)
        await message.answer(
            "Привет! Я генерирую посты.\n"
            "Команды:\n"
            "/post <тема> — сгенерировать пост\n"
            "/style <текст> — задать стиль письма\n"
            "/len short|medium|long — настроить длину\n"
            "/regen — повторить последнюю тему"
        )

    @dp.message(Command("style"))
    async def handle_style(message: types.Message) -> None:
        log_message(message)
        args = extract_args(message)
        if not args:
            await message.answer("Укажите стиль: /style <текст>")
            return
        await store.set_style(message.chat.id, args)
        await message.answer("Стиль обновлён.")

    @dp.message(Command("len"))
    async def handle_len(message: types.Message) -> None:
        log_message(message)
        args = extract_args(message)
        if not args:
            await message.answer("Укажите длину: /len short|medium|long")
            return
        length = args.lower()
        if length not in LENGTH_OPTIONS:
            await message.answer("Допустимые значения: short, medium, long.")
            return
        await store.set_length(message.chat.id, length)
        await message.answer(f"Длина установлена: {length}.")

    async def process_post(message: types.Message, topic: str) -> None:
        log_message(message)
        settings = await store.get(message.chat.id)
        try:
            content = await generate_post(
                client, topic, settings["length"], settings["style"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Generation error")
            await message.answer(f"Не удалось сгенерировать пост: {exc}")
            return
        await store.set_last_topic(message.chat.id, topic)
        for chunk in split_chunks(content):
            await message.answer(chunk)

    @dp.message(Command("post"))
    async def handle_post(message: types.Message) -> None:
        args = extract_args(message)
        if not args:
            await message.answer("Укажите тему: /post <тема>")
            return
        await process_post(message, args)

    @dp.message(Command("regen"))
    async def handle_regen(message: types.Message) -> None:
        log_message(message)
        settings = await store.get(message.chat.id)
        topic = settings.get("last_topic")
        if not topic:
            await message.answer("Нет сохранённой темы. Используйте /post <тема>.")
            return
        await process_post(message, topic)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
