# softfab/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from ..llm_client import LLMClient


class BaseAgent(ABC):
    """
    Базовый агент. Знает про LLM-клиент, но не про конкретную модель.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        task: произвольный словарь с постановкой задачи.
        Возвращает словарь-результат.
        """
        raise NotImplementedError

    def system_message(self) -> str:
        """
        Можно переопределить для конкретного агента.
        """
        return "You are a helpful assistant."

