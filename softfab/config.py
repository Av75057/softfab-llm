# softfab/config.py
import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:9000/v1"),
        api_key=os.getenv("LLM_API_KEY", "secret123"),
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
    )

