# softfab/llm_client.py
from typing import List, Dict, Any, Optional

from openai import OpenAI
from .config import load_llm_config


class LLMClient:
    def __init__(self):
        cfg = load_llm_config()
        self.model = cfg.model
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        messages: список сообщений [{role, content}] в OpenAI-формате.
        Возвращает только текст ответа (для простоты).
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            params.update(extra)

        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content

    def raw_chat(self, **kwargs) -> Any:
        """
        Если нужно получить весь raw-ответ vLLM/OpenAI.
        """
        params = {"model": self.model}
        params.update(kwargs)
        return self.client.chat.completions.create(**params)

