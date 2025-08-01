from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline.graph import chain

router = APIRouter(prefix='/api', tags=['question'])


class QuestionRequest(BaseModel):
    """
    Модель для запроса вопроса к API.

    Attributes:
        question (str): Текст вопроса, который будет передан в цепочку обработки.
    """
    question: str = Field(..., min_length=1, description='Текст вопроса для обработки')


@router.post('/ask', response_model=dict)
async def ask_question(query: QuestionRequest) -> dict:
    """
    Обрабатывает вопрос пользователя, передавая его в цепочку RAG.

    Аргументы:
        query (QuestionRequest): Объект с полем question, содержащим текст вопроса.

    Возвращает:
        dict: Словарь с ключом 'answer', содержащий ответ от цепочки обработки.

    Исключения:
        HTTPException: Если произошла ошибка при обработке вопроса
            (400 для некорректного ввода, 500 для внутренних ошибок).
    """
    try:
        # Передаем вопрос в асинхронную цепочку обработки
        result = await chain.ainvoke({'user_input': query.question})
        return {'answer': result['answer']}

    except ValueError as ve:
        # Ошибки валидации или некорректные данные
        raise HTTPException(status_code=400, detail=f'Некорректный запрос: {str(ve)}')
    except Exception as e:
        # Общие ошибки цепочки обработки
        raise HTTPException(status_code=500, detail=f'Ошибка обработки: {str(e)}')