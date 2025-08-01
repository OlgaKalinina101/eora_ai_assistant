from typing import Iterator, Set
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re


class WebTextProcessor:
    """Класс для извлечения и очистки текста с веб-страниц."""

    def __init__(self):
        # Часто встречающиеся служебные или маркетинговые фразы
        self._skip_exact: Set[str] = {
            'И наши менеджеры ответят на ваши вопросы',
            'Нажимая на кнопку, вы соглашаетесь с нашей',
            'Навыки для голосовых ассистентов',
            'Суфлекс - ИИ подсказки для операторов КЦ',
            'Викуля - ИИ поиск по базе знаний',
            'XinData - ИИ ассистент для финансовых вопросов',
            'Находясь на сайте вы соглашаетесь с применением данных технологий',
            'Топ 4 профессии, которые заменит GPT-4',
            '5 преимуществ голосового ассистента Маруся',
            'Политикой в отношении обработки',
            "Сообщение об успешной отправке",
            "Сообщение об отправке!",
            "Сообщение отправлено"
        }

        # Фразы, если они встречаются внутри строки
        self._skip_contains: Set[str] = {
            'нажимая на кнопку', 'персональных данных', 'cookies', '2025 ©', 'пожалуйста',
            'форма', 'submit', 'email', 'telegram', 'телефон',
            'голосовой ассистент', 'gpt-4', 'марус', 'соглашаетесь с'
        }

        # Фразы, с которых не должны начинаться строки
        self._skip_starts: Set[str] = {'email', 'submit', 'телефон', 'форма', 'контакты'}

    def extract_text(self, url: str) -> str:
        """
        Извлекает текст с веб-страницы по указанному URL с использованием Playwright.

        Аргументы:
            url (str): URL веб-страницы для извлечения текста.

        Возвращает:
            str: Очищенный текст страницы, разделенный переносами строк.

        Исключения:
            RuntimeError: Если не удается запустить браузер, получить содержимое или страница вернула ошибку.
        """
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page()

                # Получаем response и проверяем статус
                response = page.goto(url)
                if response.status >= 400:
                    raise RuntimeError(f"Ошибка HTTP {response.status} при обращении к {url}")

                page.wait_for_timeout(3000)  # Ожидание рендеринга (3 секунды)

                # Извлекаем текст, исключая ненужные теги
                text = page.evaluate("""
                    () => {
                        document.querySelectorAll('script, style, noscript')
                            .forEach(el => el.remove());
                        return document.body.innerText;
                    }
                """)
                browser.close()

                return '\n'.join(line.strip() for line in text.splitlines() if line.strip())

        except Exception as e:
            raise RuntimeError(f'Ошибка при извлечении текста с {url}: {str(e)}')

    def clean_html(self, html: str) -> Iterator[str]:
        """
        Очищает HTML-контент от ненужных тегов и возвращает отфильтрованный текст построчно.

        Аргументы:
            html (str): HTML-контент для обработки.

        Возвращает:
            Iterator[str]: Итератор по очищенным строкам текста.

        Исключения:
            ValueError: Если входной HTML пустой или не является строкой.
        """
        if not html or not isinstance(html, str):
            raise ValueError('Входной HTML должен быть непустой строкой')

        soup = BeautifulSoup(html, 'html.parser')
        try:
            # Удаляем ненужные теги
            for tag in soup(['script', 'style', 'nav', 'form', 'footer', 'header']):
                tag.decompose()

            # Извлекаем текст
            text = soup.get_text(separator='\n')
            soup.decompose()  # Освобождаем память

            # Заменяем неразрывные пробелы
            text = text.replace('\u00A0', ' ').replace('&nbsp;', ' ')

            # Фильтруем строки с помощью генератора
            for line in text.splitlines():
                line = line.strip()
                if len(line) > 30 and not any(
                    line.lower().startswith(word) for word in self._skip_starts
                ):
                    yield line
        finally:
            soup.decompose()  # Гарантируем освобождение памяти

    def clean_text(self, text: str) -> Iterator[str]:
        """
        Очищает текстовый блок от служебных и маркетинговых фраз.

        Аргументы:
            text (str): Текст для очистки.

        Возвращает:
            Iterator[str]: Итератор по очищенным строкам текста.

        Исключения:
            ValueError: Если входной текст пустой или не является строкой.
        """
        if not text or not isinstance(text, str):
            raise ValueError('Входной текст должен быть непустой строкой')

        for line in text.splitlines():
            line = line.strip()
            if not line or re.match(r'^\{.*\}$', line) or 'lid' in line or 'ti_name' in line:
                continue
            if line in self._skip_exact or any(sub in line.lower() for sub in self._skip_contains):
                continue
            yield line

    def process_url(self, url: str) -> str | None:
        """
        Обрабатывает URL, извлекая и очищая текст с веб-страницы.

        Аргументы:
            url (str): URL веб-страницы для обработки.

        Возвращает:
            str | None: Очищенный текст или None в случае ошибки.
        """
        try:
            raw_text = self.extract_text(url)
            clean_html_lines = self.clean_html(raw_text)
            clean_text_lines = self.clean_text('\n'.join(clean_html_lines))
            return '\n'.join(clean_text_lines)
        except Exception as e:
            print(f'Ошибка при обработке {url}: {str(e)}')
            return None


