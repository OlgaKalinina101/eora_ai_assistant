import re
import warnings
from typing import List, Set, Dict

from chromadb.api.models import Collection
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from data_extraction.dataset_builder import build_cases_dataset
from data_ingestion.ingestor import KnowledgeBaseBuilder
from settings import settings
from utils.chroma_client import get_chroma_client
from utils.logger import setup_logger

# Игнорирование предупреждения torch
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`encoder_attention_mask` is deprecated",
)

# Инициализация логгера
logger = setup_logger("chunks")

def find_relevant_chunks(
    question: str,
    collection: Collection,
    embedder: SentenceTransformer,
    top_k: int = 10,
) -> List[str]:
    """
    Семантический поиск релевантных чанков по вопросу пользователя.

    Args:
        question: Сегмент (например, "Что вы можете сделать для ритейлеров?").
        collection: Коллекция ChromaDB.
        embedder: Модель эмбеддингов (SentenceTransformer).
        top_k: Сколько самых похожих чанков вернуть.

    Returns:
        Список релевантных чанков.
    """
    # Проверка входных данных
    if not question.strip():
        logger.warning("Пустой сегмент для поиска, возвращается пустой список.")
        return []
    if top_k <= 0:
        logger.warning(f"Недопустимое значение top_k ({top_k}), возвращается пустой список.")
        return []

    try:
        # Проверка: коллекция существует, но пуста
        if collection.count() == 0:
            logger.warning("🔄 Коллекция Chroma пуста. Запускаю пересборку базы...")

            # Шаг 1: Распаковка данных
            build_cases_dataset(settings.PDF_PATH, settings.OUTPUT_JSON)
            logger.info("🔄 Векторизация источников...")

            # Шаг 2: Построение базы знаний
            builder = KnowledgeBaseBuilder()
            builder.ingest()
            logger.info("✅ База знаний успешно создана.")

            # Пересоздаем collection, чтобы она увидела изменения
            collection = get_chroma_client().get_or_create_collection(settings.CHROMA_COLLECTION_NAME)

        # Создание эмбеддинга и поиск
        query_embedding = embedder.encode(question, normalize_embeddings=True)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        # Извлекаем документы и метаданные
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results["distances"][0]

        # Склеиваем текст, source и фильтруем по расстоянию
        max_distance = 1.3  # можно сделать настраиваемым через settings
        chunks_with_sources = [
            {"text": doc, "source": meta.get("source", "unknown")}
            for doc, meta, dist in zip(documents, metadatas, distances)
            if dist <= max_distance
        ]

        filtered_chunks = rerank_by_tfidf(chunks_with_sources, question)
        logger.info(f"🔎 Найдено {len(filtered_chunks)} чанков по сегменту '{question}' (семантический поиск).")

        return filtered_chunks

    except Exception as e:
        logger.error(f"❌ Ошибка при семантическом поиске: {e}")
        return []


def rerank_by_tfidf(
    filtered_chunks: List[Dict[str, str]], question: str, top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Переранжирует текстовые фрагменты по релевантности запросу с использованием TF-IDF.

    Аргументы:
        filtered_chunks (List[Dict[str, str]]): Список словарей с текстовыми фрагментами,
            каждый содержит ключ 'text' с текстом.
        question (str): Текст запроса для оценки релевантности.
        top_k (int, optional): Количество возвращаемых наиболее релевантных фрагментов.
            По умолчанию 3.

    Возвращает:
        List[Dict[str, str]]: Список топ-k фрагментов, отсортированных по убыванию
            релевантности.

    Исключения:
        ValueError: Если входные данные некорректны (пустой список фрагментов,
            пустой запрос, некорректный top_k).
    """
    if not filtered_chunks or not all(isinstance(chunk, dict) and 'text' in chunk for chunk in filtered_chunks):
        raise ValueError('filtered_chunks должен быть непустым списком словарей с ключом "text"')
    if not question or not isinstance(question, str):
        raise ValueError('question должен быть непустой строкой')
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k должен быть положительным целым числом')

    # Инициализируем TF-IDF векторизатор
    try:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(
            chunk['text'] for chunk in filtered_chunks
        )
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        raise ValueError(f'Ошибка при вычислении TF-IDF: {str(e)}')

    # Токенизируем запрос
    query_keywords: Set[str] = set(re.findall(r'\w+', question.lower()))

    # Оцениваем релевантность каждого фрагмента
    scored_docs: List[tuple[float, Dict[str, str]]] = []
    for idx, doc in enumerate(filtered_chunks):
        tfidf_vector = tfidf_matrix[idx].toarray()[0]
        # Вычисляем веса для терминов, присутствующих в запросе
        score = sum(
            tfidf_vector[i]
            for i in tfidf_vector.nonzero()[0]
            if feature_names[i] in query_keywords
        )
        if score > 0:
            scored_docs.append((score, doc))

    # Сортируем по убыванию веса и возвращаем топ-k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]



