# PostBot

Телеграм-бот для генерации постов через OpenAI-совместимый endpoint. Настройки чата (стиль, длина, последняя тема) хранятся в SQLite.

## Команды
- `/post <тема>` — сгенерировать пост.
- `/style <текст>` — задать стиль письма для чата.
- `/len short|medium|long` — выбрать длину постов.
- `/regen` — сгенерировать заново по последней теме.

## Быстрый старт (Docker)
1. Скопируйте `.env.example` в `.env` и заполните значения.
2. Запустите:
   ```bash
   docker compose -f compose.postbot.yml up --build -d
   ```
   Если во время сборки есть проблемы с DNS/доступом к PyPI, попробуйте явно задать сеть для сборки:
   ```bash
   docker compose -f compose.postbot.yml build --build-arg PIP_INDEX_URL=https://pypi.org/simple --build-arg PIP_DEFAULT_TIMEOUT=120 --parallel
   docker compose -f compose.postbot.yml up -d
   ```
   или выполните `docker compose -f compose.postbot.yml build --build --progress=plain --no-cache` с флагом `--network host` (в compose уже указано).
3. Логи:
   ```bash
   docker compose -f compose.postbot.yml logs -f
   ```

## Переменные окружения
Смотрите `.env.example`. Важно:
- `BOT_TOKEN` — токен Telegram-бота.
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` — ваш OpenAI-совместимый endpoint.
- `POSTBOT_DB_PATH` — путь к файлу SQLite (по умолчанию `/data/postbot.db`), папка `data/` примонтирована в контейнер.
- `TELEGRAM_PROXY` — опционально, HTTP(S)-прокси для Telegram (например, `http://user:pass@host:port`), если прямой доступ закрыт.
- `TELEGRAM_TIMEOUT` — таймаут запросов к Telegram API в секундах (используется в том случае, если вы включите прокси через переменную ниже; для простоты по умолчанию не применяется явный timeout-объект).

### Хранилище
По умолчанию использован именованный volume `postbot_data`, монтируется в `/data`. SQLite лежит в `/data/postbot.db`. Чтобы использовать bind-монтаж (например, для просмотра файла на хосте), замените в `compose.postbot.yml` строку `- postbot_data:/data` на `- ./data:/data` и убедитесь, что директория `data/` существует.

### Сеть
Сервис запущен с `network_mode: host`, чтобы обойти возможные блокировки/DNS-проблемы при обращении к Telegram. Если в вашей среде это нежелательно, удалите `network_mode: host` и при необходимости задайте `TELEGRAM_PROXY`.

## Локальный запуск без Docker
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заполните значения
python main.py
```
