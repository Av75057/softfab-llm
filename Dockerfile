FROM python:3.12-slim

WORKDIR /app

# Копируем только main.py
COPY main.py .

# Ставим зависимости
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx

# Настройки окружения по умолчанию
ENV VLLM_URL=http://vllm:8000/v1
ENV API_KEY=
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
