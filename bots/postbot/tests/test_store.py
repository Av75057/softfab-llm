import asyncio
from pathlib import Path

from main import ChatSettingsStore


def test_store_roundtrip(tmp_path: Path) -> None:
    async def run_flow() -> None:
        db_path = tmp_path / "postbot.db"
        store = ChatSettingsStore(str(db_path))
        await store.init()

        defaults = await store.get(chat_id=1)
        assert defaults["length"] == "medium"
        assert defaults["style"] is None
        assert defaults["last_topic"] is None

        await store.set_style(1, "дружелюбно")
        await store.set_length(1, "long")
        await store.set_last_topic(1, "Искусственный интеллект")

        saved = await store.get(chat_id=1)
        assert saved["style"] == "дружелюбно"
        assert saved["length"] == "long"
        assert saved["last_topic"] == "Искусственный интеллект"

    asyncio.run(run_flow())


def test_init_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "postbot.db"
    store = ChatSettingsStore(str(db_path))
    asyncio.run(store.init())
    asyncio.run(store.init())  # Should not raise
