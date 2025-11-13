from openai import OpenAI
from softfab.config import load_llm_config

def main():
    cfg = load_llm_config()
    client = OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )

    print("[STREAM TEST] Отправляю stream-запрос...")
    stream = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": "Ты отвечаешь коротко и по делу."},
            {"role": "user", "content": "Сгенерируй 1-2 предложения, я хочу увидеть поток символов."},
        ],
        max_tokens=128,
        temperature=0.3,
        stream=True,
    )

    print("[STREAM TEST] Ответ (онлайн):")
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    print("\n[STREAM TEST] Готово.")

if __name__ == "__main__":
    main()

