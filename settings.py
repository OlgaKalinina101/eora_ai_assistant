import os

from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Загружаем .env, если он есть

class Settings(BaseSettings):
    # Абсолютный путь до корня проекта
    BASE_DIR: Path = Path(__file__).resolve().parent

    # Пути к файлам
    PDF_PATH: Path = BASE_DIR / "data" / "raw" / "Тестовое задание EORA Разработчик.pdf"
    OUTPUT_JSON: Path = BASE_DIR / "data" / "eora_cases.json"

    # Пути к базе данных
    CHROMA_DB_PATH: Path = BASE_DIR / "vector_store"
    CHROMA_COLLECTION_NAME: str = "eora_cases"

    # Название модели эмбеддингов (с возможностью переопределить через .env)
    EMBEDDING_MODEL_NAME: str = "sberbank-ai/sbert_large_nlu_ru"

    CHUNK_SIZE: int = 150
    CHUNK_OVERLAP: int = 30

    #OpenAI API_KEY
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    class Config:
        env_file = ".env"  # ← если захочешь переопределять из файла окружения


settings = Settings()
