from typing import TypedDict, List

class Chunk(TypedDict):
    """
    Фрагмент текста с метаданными для обработки и ранжирования.

    Используется для типизации словарей, содержащих текст и источник.

    Attributes:
        text (str): Текст фрагмента.
        source (str): URL или другой идентификатор источника текста.
    """
    text: str
    source: str

class LetterState(TypedDict):
    """
    Типизированное состояние для конвейера генерации ответа.

    Attributes:
        user_input: Вопрос пользователя.
        chunks: Список релевантных чанков из базы знаний.
        prompt: Промпт для генерации ответа на вопрос.
        answer: Сгенерированный ответ."""
    user_input: str
    chunks: List[Chunk]
    prompt: str
    answer: str