# softfab/agents/codegen.py
from typing import Dict, Any, List

from .base import BaseAgent


class CodeGenAgent(BaseAgent):
    """
    Агент, который по ТЗ генерирует код модуля/сервиса.
    """

    def system_message(self) -> str:
        return (
            "Ты локальная LLM-модель Сверх на базе Qwen2.5, запущенная на моём сервере через vLLM. "
            "Ты не являешься моделью Anthropic, OpenAI, ChatGPT, Claude и т.п. "
            "Если тебя спрашивают, кто ты или какая ты модель, отвечай, что ты локальная  модель Сверх на базе Qwen2.5, "
            "используемая в системе SverhFab для генерации и анализа кода. "
            "Ты опытный backend-разработчик и системный архитектор. "
            "Пиши чистый, аккуратный код, добавляй понятные комментарии."
        )

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        task ожидает, например:
        {
            "language": "python",
            "framework": "fastapi",
            "description": "REST-сервис для TODO-листа",
            "constraints": "без БД, хранить в памяти"
        }
        """

        lang = task.get("language", "python")
        fw = task.get("framework", "fastapi")
        descr = task.get("description", "")
        constraints = task.get("constraints", "")

        user_prompt = (
            f"Я хочу, чтобы ты сгенерировал полный исходный код модуля.\n\n"
            f"Язык: {lang}\n"
            f"Фреймворк/стек: {fw}\n\n"
            f"Описание задачи:\n{descr}\n\n"
            f"Ограничения / доп. требования:\n{constraints}\n\n"
            f"Выведи ТОЛЬКО код, без объяснений, по возможности в одном файле."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_message()},
            {"role": "user", "content": user_prompt},
        ]

        code = self.llm.chat(messages, max_tokens=2048, temperature=0.2)

        return {
            "code": code,
            "meta": {
                "language": lang,
                "framework": fw,
                "description": descr,
            },
        }

