import os
from pathlib import Path
from typing import List
from rag.pipeline.types import Chunk

# Формируем путь к файлу шаблона относительно текущего скрипта
PROMPT_PATH = Path(os.path.join(os.path.dirname(__file__), 'prompt_template.txt'))


def load_prompt_template(prompt_path: Path = PROMPT_PATH) -> str:
    """
    Загружает шаблон промпта из файла по указанному пути.

    Аргументы:
        prompt_path (Path, optional): Путь к файлу с шаблоном промпта.
            По умолчанию используется PROMPT_PATH, сформированный относительно
            текущего скрипта.

    Возвращает:
        str: Содержимое файла шаблона промпта.

    Исключения:
        FileNotFoundError: Если файл не существует.
        ValueError: Если путь не указывает на файл с текстовым содержимым.
    """
    if not isinstance(prompt_path, Path):
        prompt_path = Path(prompt_path)

    if not prompt_path.exists():
        raise FileNotFoundError(f'Файл {prompt_path} не найден')
    if not prompt_path.is_file():
        raise ValueError(f'{prompt_path} должен быть файлом')

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f'Ошибка при чтении файла {prompt_path}: {str(e)}')


def build_context(docs: List[Chunk]) -> str:
    """
    Формирует контекст из списка текстовых фрагментов в формате с нумерацией и источниками.

    Аргументы:
        docs (List[Chunk]): Список словарей с ключами 'text' и 'source'.

    Возвращает:
        str: Отформатированный контекст, где каждый фрагмент нумеруется и включает источник.

    Исключения:
        ValueError: Если docs пустой или содержит некорректные элементы.
    """
    if not docs or not all(isinstance(doc, dict) and 'text' in doc and 'source' in doc for doc in docs):
        raise ValueError('docs должен быть непустым списком словарей с ключами "text" и "source"')

    # Формируем строки с помощью генератора для экономии памяти
    formatted_lines = (
        f"[{i}] {doc['text'].strip()}\nИсточник: {doc['source']}"
        for i, doc in enumerate(docs, 1)
    )
    return '\n\n'.join(formatted_lines)


def attach_links(answer: str, docs: List[Chunk]) -> str:
    """
    Добавляет Markdown-ссылки в ответ, заменяя нумерацию [i] на [i](source).

    Аргументы:
        answer (str): Текст ответа, содержащий нумерацию вида [i].
        docs (List[Chunk]): Список словарей с ключами 'text' и 'source'.

    Возвращает:
        str: Ответ с замененными номерами на Markdown-ссылки.

    Исключения:
        ValueError: Если docs содержит некорректные элементы или answer не строка.
    """
    if not isinstance(answer, str):
        raise ValueError('answer должен быть строкой')
    if not all(isinstance(doc, dict) and 'source' in doc for doc in docs):
        raise ValueError('docs должен содержать словари с ключом "source"')

    result = answer
    for i, doc in enumerate(docs, 1):
        source = doc.get('source', '').strip()
        if source:
            result = result.replace(f'[{i}]', f'[{i}]({source})')

    return result