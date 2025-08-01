# 🧠 Ассистент EORA

Прототип системы с retrieval-augmented generation (RAG) на базе LangGraph и OpenAI для предоставления точных и контекстных ответов потенциальным клиентам компании EORA, используя кейсы и материалы с сайта eora.ru.

## 📝 Обзор проекта

Цель проекта — разработка прототипа ИИ-ассистента, который отвечает на вопросы клиентов, опираясь на внутренние кейсы и контент сайта eora.ru. Реализация выполнена на высоком уровне сложности и включает:

- Ответ содержит примеры из кейсов.
- Источники оформлены в формате [1], [2], как в PDF.

## ✅ Достижения

Прототип успешно реализует устойчивый пайплайн на базе LangGraph, включающий:

- Извлечение URL-адресов из PDF и дополнение их парсингом сайта с использованием Playwright.
- Создание векторной базы знаний с помощью ChromaDB и SentenceTransformers для семантического поиска.
- Реализацию семантического переранжирования с использованием TF-IDF для выбора top-k релевантных фрагментов.
- Генерацию профессиональных ответов на вопросы с примерами, оформленных для общения с клиентами.

## 📂 Структура проекта

```
EORA_AI_assistant_rag/
├── api/                        # Основная логика приложения
│   ├── __init__.py
│   └── endpoints.py            # Маршруты FastAPI
├── data/                       # Хранилище данных
│   ├── eora_cases.json         # Данные кейсов
│   └── raw/                    # Исходные файлы
│       └── Тестовое задание EORA Разработчик.pdf
├── data_extraction/            # Утилиты для извлечения данных
│   ├── __init__.py
│   ├── dataset_builder.py      # Пайплайн обработки HTML
│   ├── pdf_extractor.py        # Извлечение URL из PDF
│   └── web_processor.py        # WebTextProcessor для загрузки и очистки контента
├── data_ingestion/             # Пайплайн обработки документов
│   ├── __init__.py
│   ├── ingestor.py             # Оркестрация пайплайна
│   └── loader.py               # Загрузка данных 
├── rag/                        # Пайплайн RAG
│   └── pipeline/               # Логика LangGraph
│       ├── __init__.py
│       ├── graph.py            # Сборка рабочего процесса LangGraph
│       ├── chunk_selector.py   # Поиск релевантных фрагментов в Chroma
│       ├── nodes.py            # Узлы LangGraph
│       ├── prompt_template.txt # Контекстный промпт для модели
│       └── types.py            # Типы состояния пайплайна
│   ├── __init__.py
│   └── openai_client.py        # Конфигурация клиента OpenAI
├── utils/                      # Вспомогательные утилиты
│   ├── __init__.py
│   ├── logger.py               # Настройка логирования
│   └── chroma_client.py        # Конфигурация клиента ChromaDB
├── vector_store/               # Векторная база данных
├── settings.py                 # Конфигурация путей, БД и моделей
├── main.py                     # Точка входа приложения FastAPI
├── requirements.txt            # Зависимости проекта
├── .env                        # Переменные окружения
└── README.md                   # Документация проекта
```

## ⚙️ Пайплайн обработки данных

Функция `build_cases_dataset` выполняет следующие шаги:

1. **Извлечение URL**: Извлекает URL-адреса из PDF с помощью `extract_urls_from_pdf`.
2. **Парсинг и очистка**: Для каждого URL:
   - Извлекается контент страницы и удаляются ненужные теги (например, скрипты, стили) с помощью `WebTextProcessor`.
   - Очищенные данные объединяются в строку с использованием `'\n'.join(...)` для оптимизации памяти.
3. **Сохранение результата**: Итоговые данные записываются в JSON-файл (`eora_cases.json`) с кодировкой UTF-8.

### Рекомендации по оптимизации
- **Асинхронная обработка**: Переписать `build_cases_dataset` с использованием `asyncio` и асинхронного Playwright для параллельной обработки URL, сокращая время выполнения.
- **Управление памятью**: Использовать `memory_profiler` для анализа и оптимизации потребления памяти при обработке больших данных.
- **Кэширование**: Реализовать кэширование результатов обработки URL (например, через `functools.lru_cache`) для повышения производительности.
- **Тестирование**: Добавить модульные тесты с `pytest` для проверки корректности обработки итераторов и кодировки.

## 🛠️ Класс WebTextProcessor

Класс `WebTextProcessor` отвечает за извлечение и очистку веб-контента. Пример использования:

```python
processor = WebTextProcessor()
urls = extract_urls_from_pdf(pdf_path)
```

### Основные возможности
- **Инициализация**: Создает экземпляр `WebTextProcessor` с наборами фильтров для удаления тегов.
- **Обработка**: Метод `process_url` последовательно вызывает `extract_text`, `clean_html` и `clean_text`.
- **Эффективное извлечение**: Использует `page.evaluate` из Playwright для удаления ненужных тегов (например, `<script>`, `<style>`, `<noscript>`) в браузере, минимизируя объем данных, передаваемых в Python, и устраняя необходимость в BeautifulSoup.
- **Оптимизация памяти**: Методы `clean_html_text` и `clean_text_block` возвращают итераторы, избегая создания промежуточных списков. Финальная сборка строк выполняется через `'\n'.join(...)`.
- **Управление ресурсами**: Гарантирует освобождение ресурсов браузера и объектов.

### Рекомендации по оптимизации
- **Асинхронный Playwright**: Перейти на `async_playwright` для параллельной обработки URL.
- **Тестирование**: Реализовать тесты с `pytest` для проверки обработки URL, HTML и текста.
- **Кэширование**: Добавить `functools.lru_cache` в `process_url` для повторной обработки одних и тех же URL.

### Хранение метаданных
- Сохраняет URL источника для отслеживания.

## 📚 Пайплайн загрузки данных

Пайплайн загрузки данных отвечает за структурированную индексацию кейсов в ChromaDB для семантического поиска.

### Шаги выполнения
```python
builder = KnowledgeBaseBuilder()
builder.ingest()
```

### Модель эмбеддингов
- **Используемая модель**: `sberbank-ai/sbert_large_nlu_ru`
- **Преимущества**: Оптимизирована для русскоязычного контента, подходит для деловых текстов и семантического поиска.
- **Альтернативные модели**:

  | Модель                        | Преимущества                                         | Недостатки                     |
  |-------------------------------|------------------------------------------------------|--------------------------------|
  | multi-qa-mpnet-base-dot-v1    | Высокая точность для Q&A, хороша для длинных текстов | Медленнее, больше памяти       |
  | BAAI/bge-small-en             | Баланс качества и скорости                           | Требует ручной нормализации    |
  | intfloat/e5-small-v2          | Подходит для open-domain retrieval                   | Требует формата query/passage  |
  | text-embedding-3-small        | Высокое качество эмбеддингов                         | Платная, требует API           |

### 📐 Обоснование параметров CHUNK_SIZE и CHUNK_OVERLAP  
Для эффективной генерации ответов LLM-моделью было выбрано:  

```python
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
```

Почему такие значения?

CHUNK_SIZE = 150 — тексты, предоставленные в кейсах (например, https://eora.ru/cases/skazki-dlya-gugl-assistenta), представляют собой плотные фрагменты с перемежающимися фактами, решениями и деталями проектов. Размер 150 токенов (примерно 100–120 слов) позволяет сохранить цельный смысловой блок (например, описание одного этапа проекта), не дробя его слишком мелко. Это важно, чтобы при генерации ответов LLM могла опираться на связный фрагмент, а не отдельные обрывки.

CHUNK_OVERLAP = 30 — перекрытие используется для сглаживания границ между чанками. Это особенно полезно, когда важная информация попадает на стык двух фрагментов. Такое перекрытие (~20% от длины чанка) помогает избежать потери контекста при поиске и генерации.

Вывод:
Такая стратегия чанкинга обеспечивает семантическую целостность фрагментов, повышает релевантность семантического поиска и улучшает согласованность ответов, особенно в условиях коротких и насыщенных материалов, как в задании EORA.

### Архитектура пайплайна

| Этап                         | Описание                                                                                                                   |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **1. Загрузка данных**       | Чтение кейсов из `eora_cases.json` в формате `{link: text}`, нормализация текста, преобразование в `llama_index.Document`. |
| **2. Разбиение на чанки**    | Деление документов на смысловые блоки с помощью `SentenceSplitter` с настраиваемыми `chunk_size` и `chunk_overlap`.        |
| **3. Генерация эмбеддингов** | Использование `SentenceTransformer` (`sberbank-ai/sbert_large_nlu_ru`) для создания эмбеддингов чанков.                    |
| **4. Пакетная обработка**    | Обработка чанков пакетами по 100 для оптимизации CPU и памяти.                                                             |
| **5. Сохранение в ChromaDB** | Добавление документов, эмбеддингов и метаданных (`source` URL) в коллекцию ChromaDB.                                       |
| **6. Мониторинг памяти**     | Логирование потребления RAM через `psutil`.                                                                                |

### Технологии и инструменты
- **Представление документов**: `llama_index.Document` для структурированных данных с метаданными.
- **Разбиение на чанки**: `SentenceSplitter` для разделения по предложениям с перекрытием.
- **Эмбеддинги**: `sentence-transformers` с моделью `sberbank-ai/sbert_large_nlu_ru`.
- **Хранилище**: `ChromaDB` для векторного хранения и поиска.
- **Мониторинг**: `psutil` для отслеживания потребления памяти.
- **Логирование**: Кастомный логгер для записи этапов обработки, ошибок и результатов.

### Оптимизации
- **Пакетная обработка**: Генерация эмбеддингов пакетами по 100 для снижения нагрузки на CPU и память.
- **Очистка памяти**: Явное освобождение памяти после каждого пакета через `del`.
- **Логирование памяти**: Отслеживание Resident Set Size (RSS) после каждого вызова `encode()`.
- **Обработка ошибок**: Многоуровневая обработка ошибок (`try/except`) на уровне документа и пакета.

## 🧠 Пайплайн LangGraph

Пайплайн построен на LangGraph и следует принципу единственной ответственности (SRP) с разделением на узлы:

| Узел            | Назначение                                      |
|-----------------|-------------------------------------------------|
| `input`         | Принимает пользовательский ввод (`user_input`). |
| `search`        | Находит релевантные чанки по сегменту.          |
| `prompt`        | Формирует промпт для LLM.                       |
| `generate`      | Генерирует письмо через `gpt-4o`.               |
| `output`        | Возвращает финальное письмо.                    |

### Оптимизация памяти
- Повторное использование глобальных ресурсов (`SentenceTransformer`, ChromaDB, `AsyncOpenAI`).
- Валидация данных на каждом узле.
- Мониторинг памяти с помощью `psutil` в узлах `search` и `generate`.

## 🧠 Инжиниринг промптов

Для генерации выбрана модель `gpt-4o` за оптимальное соотношение цены, качества и предсказуемости ответа. Конфигурация:

```python
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
```

- **Температура**: Установлена на 0.7 для создания креативных, но контролируемых писем.
- **Дизайн промпта**: Использует контекстно-ориентированный подход RAG, включая релевантные выдержки из кейсов EORA, чтобы:
  - Исключить галлюцинации, опираясь на конкретные источники.
  - Повысить точность и релевантность.
  - Обеспечить прозрачность с указанием источников.
- **Расширяемость**: Легко адаптируется к новым сегментам или продуктам путем замены контекста.

## 🌐 HTTP API

Сервис доступен через POST-запрос на эндпоинт `/ask`.

### Пример запроса
```json
POST /ask
{
  "question": "Что вы можете сделать для ритейлеров?"
}
```

### Пример ответа
```json
{
  "answer": "Уважаемый(ая) [Получатель],\n\nДля ритейлеров компания EORA предлагает решения, направленные на повышение вовлеченности клиентов и оптимизацию процессов:\n\n1. **Интерактивные навыки и боты**: Мы разработали викторину для Purina на платформе Алиса, которая повысила доверие клиентов. Это решение может помочь ритейлерам укрепить лояльность к бренду [1](https://eora.ru/cases/purina-navyk-viktorina).\n\n2. **Автоматизация HR-процессов**: Для компании Магнит мы создали HR-бота, упрощающего приглашение кандидатов на собеседования, что оптимизирует процесс найма [2](https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie).\n\nМы будем рады обсудить ваши задачи и предложить индивидуальные решения.\n\nЕсли у вас есть вопросы или вы хотите узнать о других кейсах, пожалуйста, свяжитесь с нами!\n\nС уважением,\n[Ваше имя]"
}
```

## 🚀 Запуск проекта

### Локальный запуск
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
2. Настройте переменные окружения в `.env`:
   ```plaintext
   OPENAI_API_KEY=sk-...
   ```
3. Запустите приложение:
   ```bash
   uvicorn main:app --reload
   ```

### Запуск через Docker
1. Соберите и запустите сервисы:
   ```bash
   docker-compose up --build
   ```
2. Доступ к API: `http://localhost:8000/ask`.

## 🔧 Рекомендации по улучшению
- **Усовершенствование RAG**:
  - Добавить синонимический или fuzzy-поиск для сегментов рынка.
  - Сохранять нормализованные сегменты рынка в метаданных на этапе чанкинга.
- **Кэширование**:
  - Кэшировать результаты поиска чанков для повторяющихся запросов:
    ```python
    from functools import lru_cache
    @lru_cache(maxsize=100)
    def cached_search(segment: str) -> tuple:
        return tuple(find_relevant_chunks_by_segment(segment, collection, embedder))
    ```
- **Тестирование**:
  - Добавить интеграционные тесты для эндпоинта `/ask` с использованием `TestClient`.
- **Guardrails**:
  - Внедрить валидацию ответов модели с помощью Guardrails для проверки фактической точности, стилистической нейтральности и ограничения длины.
- **Безопасность**:
  - Реализовать аутентификацию API (например, JWT или API-ключ).
