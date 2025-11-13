# softfab/run_codegen_demo.py
from softfab.llm_client import LLMClient
from softfab.agents.codegen import CodeGenAgent


def main():
    llm = LLMClient()
    agent = CodeGenAgent(llm)

    task = {
        "language": "python",
        "framework": "fastapi",
        "description": "REST сервис с /health и /items (CRUD по in-memory списку).",
        "constraints": "без БД, без авторизации, один файл main.py"
    }

    result = agent.run(task)

    print("=== Сгенерированный код ===")
    print(result["code"])


if __name__ == "__main__":
    main()

