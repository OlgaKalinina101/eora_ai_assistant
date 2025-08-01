import json

from pathlib import Path
from typing import Generator
from llama_index.core import Document


def iterate_cases(json_path: Path) -> Generator[Document, None, None]:
    """
    Читает JSON-файл формата {link: text} и возвращает построчно кортежи с данными.

    Аргументы:
        json_path (Path): Путь к JSON-файлу с данными.

    Возвращает:
        Generator[Tuple[str, str, str], None, None]: Генератор кортежей, где каждый кортеж
            содержит:
            - URL (ключ из JSON),
            - Полный текст с удаленными переносами строк и лишними пробелами.

    Исключения:
        FileNotFoundError: Если JSON-файл не существует.
        ValueError: Если файл не является JSON или содержит некорректные данные.
    """
    if not json_path.exists():
        raise FileNotFoundError(f'Файл {json_path} не найден')
    if json_path.suffix.lower() != '.json':
        raise ValueError('Файл должен иметь расширение .json')

    try:
        with open(json_path, encoding='utf-8') as f:
            cases = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Ошибка декодирования JSON в файле {json_path}: {str(e)}')
    except Exception as e:
        raise ValueError(f'Ошибка при чтении файла {json_path}: {str(e)}')

    if not isinstance(cases, dict):
        raise ValueError('JSON должен быть объектом с парами {link: text}')

    for link, text in cases.items():
        if not isinstance(text, str):
            text = str(text)  # Приводим к строке, если текст не строка

        # Убираем переносы строк, кавычки и лишние пробелы
        clean_text = ' '.join(text.split()).translate(str.maketrans('', '')).strip()

        yield Document(
            text=clean_text,
            metadata={"source": link}
        )
