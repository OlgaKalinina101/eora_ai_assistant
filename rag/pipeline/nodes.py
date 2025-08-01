
import os
import time

import psutil
from sentence_transformers import SentenceTransformer

from rag.pipeline.chunk_selector import find_relevant_chunks
from rag.openai_client import client
from rag.pipeline.helpers import build_context, load_prompt_template, attach_links
from rag.pipeline.types import LetterState, Chunk
from settings import settings
from utils.chroma_client import get_chroma_client
from utils.logger import setup_logger

# Инициализация логгера
logger = setup_logger("letter_pipeline")

# Глобальная инициализация клиента ChromaDB и модели эмбеддингов
chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
openai_client = client

# Определение узлов конвейера
async def input_node(state: LetterState) -> LetterState:
    """
    Принимает начальное состояние и возвращает его без изменений.

    Args:
        state: Состояние конвейера с пользовательскими данными.

    Returns:
        То же состояние без изменений.
    """
    # Возврат входного состояния
    return state


async def search_chunks_node(state: LetterState) -> LetterState:
    """
    Выполняет семантический поиск релевантных чанков по сегменту.

    Args:
        state: Состояние конвейера с пользовательскими данными.

    Returns:
        Обновленное состояние с добавленным списком чанков.
    """
    # Проверка наличия и корректности сегмента
    if not isinstance(state.get("user_input"), str):
        logger.error("user_input отсутствует или некорректен.")
        return {**state, "chunks": []}

    # Извлечение сегмента и поиск чанков
    segment = state["user_input"]
    chunks = find_relevant_chunks(segment, chroma_collection, embedder)

    # Логирование потребления памяти
    logger.info(
        f"Потребление памяти после поиска чанков: "
        f"{psutil.Process().memory_info().rss / 1024**2:.2f} МБ"
    )

    # Обновление состояния с найденными чанками
    return {**state, "chunks": chunks}


async def build_prompt_node(state: LetterState) -> LetterState:
    """
    Формирует промпт для генерации письма на основе пользовательских данных и чанков.

    Args:
        state: Состояние конвейера с пользовательскими данными и чанками.

    Returns:
        Обновленное состояние с добавленным промптом.
    """
    # Проверка наличия необходимых данных
    if not isinstance(state.get("user_input"), str) or not state.get("chunks"):
        logger.error("Отсутствуют необходимые данные: user_input или chunks.")
        return {**state, "prompt": ""}

    user_input = state["user_input"]
    chunks = state["chunks"]

    # Формирование контекста из чанков
    context = build_context(chunks)

    # Создание промпта для письма
    template = load_prompt_template()

    try:
        prompt = template.format(question=user_input, chunks=context)
    except KeyError as e:
        logger.error(f"Ошибка форматирования шаблона: отсутствует ключ {e}")
        return {**state, "prompt": ""}

    # Обновление состояния с промптом
    logger.info(f"prompt: {prompt}")
    return {**state, "prompt": prompt}


async def generate_letter_node(state: LetterState) -> LetterState:
    """
    Генерирует деловое письмо с помощью OpenAI API на основе промпта.

    Args:
        state: Состояние конвейера с промптом.

    Returns:
        Обновленное состояние с сгенерированным ответом.
    """
    # Проверка наличия промпта
    if not state.get("prompt"):
        logger.error("Отсутствует промпт для генерации письма.")
        return {**state, "answer": ""}

    # Генерация письма через асинхронный OpenAI API
    try:
        start_time = time.perf_counter()

        logger.info("Отправляем запрос в OpenAI API")
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Ты — AI-эксперт по проектам компании EORA.",
                },
                {"role": "user", "content": state["prompt"]},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content if response.choices else ""
        content_with_links = attach_links(content, state["chunks"])
        elapsed = time.perf_counter() - start_time
        logger.info(f"📨 Ответ: {content_with_links}")

        # Логгирование времени генерации
        logger.info(f"📨 Ответ успешно сгенерирован за {elapsed:.2f} секунд.")

        # Логирование потребления памяти
        logger.info(
            f"Потребление памяти после генерации письма: "
            f"{psutil.Process().memory_info().rss / 1024**2:.2f} МБ"
        )

        # Обновление состояния с сгенерированным ответом
        return {**state, "answer": content_with_links}

    except Exception as e:
        logger.error(f"Ошибка при генерации письма: {e}")
        return {**state, "answer": ""}


async def output_node(state: LetterState) -> LetterState:
    """
    Возвращает состояние с сгенерированным ответом.

    Args:
        state: Состояние конвейера с ответом.

    Returns:
        Состояние с ответом (для совместимости с LangGraph).
    """
    # Проверка наличия письма
    if not state.get("answer"):
        logger.warning("Ответ отсутствует в состоянии.")
        return {**state, "answer": ""}

    # Возврат состояния для совместимости с LangGraph
    return state