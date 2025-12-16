from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite


DEFAULT_PROFILE = {
    "style": "",
    "length": "medium",
    "last_topic": None,
    "last_response": None,
}


class UserRepository:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                style TEXT DEFAULT '',
                length TEXT DEFAULT 'medium',
                last_topic TEXT,
                last_response TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def _ensure(self) -> None:
        if not self.conn:
            raise RuntimeError("Database connection is not established")

    async def _ensure_user(self, user_id: int) -> None:
        await self._ensure()
        await self.conn.execute(
            "INSERT OR IGNORE INTO user_preferences (user_id) VALUES (?)", (user_id,)
        )
        await self.conn.commit()

    async def get_profile(self, user_id: int) -> Dict[str, Any]:
        await self._ensure()
        await self._ensure_user(user_id)
        async with self.conn.execute(
            """
            SELECT style, length, last_topic, last_response
            FROM user_preferences
            WHERE user_id = ?
            """,
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return DEFAULT_PROFILE.copy()
        return {
            "style": row[0] or "",
            "length": row[1] or "medium",
            "last_topic": row[2],
            "last_response": row[3],
        }

    async def update_profile(self, user_id: int, **fields: Any) -> None:
        await self._ensure()
        await self._ensure_user(user_id)
        allowed_keys = {"style", "length", "last_topic", "last_response"}
        filtered = {k: v for k, v in fields.items() if k in allowed_keys}
        if not filtered:
            return
        set_clause = ", ".join(f"{key} = ?" for key in filtered.keys())
        values = list(filtered.values()) + [user_id]
        await self.conn.execute(
            f"""
            UPDATE user_preferences
            SET {set_clause},
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            values,
        )
        await self.conn.commit()
