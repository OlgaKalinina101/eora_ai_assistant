import json
from pathlib import Path

from tqdm import tqdm

from data_extraction.extractor import extract_urls_from_pdf
from data_extraction.web_processor import WebTextProcessor
from utils.logger import setup_logger
from settings import settings

# Инициализация логгера
logger = setup_logger("extracted")


def build_cases_dataset(pdf_path: Path, output_json: Path) -> None:
    """
    Обрабатывает URL из PDF, извлекает и очищает текст, сохраняет результаты в JSON.

    Аргументы:
        pdf_path (Path): Путь к PDF-файлу с URL.
        output_json (Path): Путь к выходному JSON-файлу.
    """

    processor = WebTextProcessor()
    urls = extract_urls_from_pdf(pdf_path)

    results: dict[str, str] = {}

    for url in tqdm(urls, desc='Обработка ссылок'):
        try:
            content = processor.process_url(url)

            # Фильтруем пустые и ошибочные кейсы
            if content and content.strip():
                results[url] = content
            else:
                logger.warning(f'Пустой или некорректный результат для {url}, пропущено')
        except Exception as e:
            logger.error(f'Ошибка при обработке {url}: {str(e)}')
            continue

    # Сохраняем в JSON
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f'Готово: сохранено {len(results)} кейсов в {output_json}')

    except Exception as e:
        logger.error(f'Ошибка при сохранении результатов в {output_json}: {str(e)}')
        raise RuntimeError(f'Не удалось сохранить результаты: {str(e)}')


if __name__ == '__main__':
    PDF_PATH = Path(settings.PDF_PATH)  # Путь к PDF
    OUTPUT_JSON = settings.OUTPUT_JSON  # Путь к JSON
    build_cases_dataset(PDF_PATH, OUTPUT_JSON)
