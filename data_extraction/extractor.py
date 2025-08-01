import re
from pathlib import Path
from typing import List

import fitz


def extract_urls_from_pdf(pdf_path: Path) -> List[str]:
    """
    Извлекает уникальные URL-адреса из PDF-документа.

    Аргументы:
        pdf_path (Path): Путь к PDF-файлу.

    Возвращает:
        List[str]: Список уникальных URL-адресов, найденных в документе.

    Исключения:
        FileNotFoundError: Если указанный PDF-файл не существует.
        ValueError: Если pdf_path не является объектом Path или файл не является PDF.
    """
    if not isinstance(pdf_path, Path):
        raise ValueError('pdf_path должен быть объектом Path')
    if not pdf_path.exists():
        raise FileNotFoundError(f'Файл {pdf_path} не найден')
    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError('Файл должен иметь расширение .pdf')

    urls: List[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                page_urls = re.findall(r'https?://[^\s)]+', text)
                urls.extend(page_urls)
        return list(set(urls))
    except Exception as e:
        raise RuntimeError(f'Ошибка при обработке PDF-файла {pdf_path}: {str(e)}')


