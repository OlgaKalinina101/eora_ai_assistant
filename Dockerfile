# Используем официальный Python-образ
FROM python:3.12-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential \
    && pip install --upgrade pip \
    && pip install --retries=10 --timeout=60 --default-timeout=60 --progress-bar off -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Копируем всё приложение
COPY . .

# Указываем переменные окружения
ENV PYTHONUNBUFFERED=1

# Открываем порт
EXPOSE 8000

# Команда запуска FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]