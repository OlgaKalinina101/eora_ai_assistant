# Настройка клиента
from openai import AsyncOpenAI

from settings import settings

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY
)