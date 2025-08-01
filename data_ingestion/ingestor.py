from typing import List
import psutil

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from sentence_transformers import SentenceTransformer

from data_ingestion.loader import iterate_cases
from settings import settings
from utils.chroma_client import get_chroma_collection, get_chroma_client
from utils.logger import setup_logger

# Инициализация логгера
logger = setup_logger("chroma")


class KnowledgeBaseBuilder:
    def __init__(self) -> None:
        """Инициализирует ChromaDB клиент и модель эмбеддингов."""

        #Инициализация клиента Chroma DB
        self.client = get_chroma_client()

        # Получение или создание коллекции ChromaDB
        self.collection = get_chroma_collection(self.client)

        # Загрузка модели для создания эмбеддингов
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    def chunk_document(self, doc: Document) -> List[Document]:
        """
        Разбивает документ на чанки фиксированного размера.

        Args:
            doc: Объект Document для разбиения.

        Returns:
            Список объектов Document, каждый из которых содержит чанк текста и метаданные.
        """
        # Инициализация разделителя текста на чанки
        splitter = SentenceSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

        # Разбиение текста на чанки
        chunks = splitter.split_text(doc.text)

        # Создание объектов Document для каждого чанка
        return [Document(text=chunk, metadata=doc.metadata) for chunk in chunks]

    def ingest(self) -> None:
        """Загружает документы в ChromaDB, разбивая их на чанки и создавая эмбеддинги."""
        # Подсчет общего количества обработанных чанков

        total_chunks = 0
        batch_size = 100  # Размер батча для обработки эмбеддингов

        # Обработка кейсов по одному через итератор
        for doc in iterate_cases(json_path=settings.OUTPUT_JSON):
            try:
                # Разбиение документа на чанки
                doc_chunks = self.chunk_document(doc)
                chunks = [chunk.text for chunk in doc_chunks]
                metadatas = [chunk.metadata for chunk in doc_chunks]

                # Обработка чанков батчами
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i : i + batch_size]
                    batch_metadatas = metadatas[i : i + batch_size]

                    # Создание эмбеддингов для батча
                    try:

                        batch_embeddings = self.embedder.encode(batch_chunks, normalize_embeddings=True)
                        logger.info(
                            f"Потребление памяти после создания эмбеддингов: "
                            f"{psutil.Process().memory_info().rss / 1024**2:.2f} МБ"
                        )
                    except Exception as e:
                        logger.error(f"Ошибка при создании эмбеддингов: {e}")
                        continue

                    # Добавление батча в ChromaDB
                    try:
                        self.collection.add(
                            documents=batch_chunks,
                            metadatas=batch_metadatas,
                            embeddings=batch_embeddings,
                            ids=[f"doc_{total_chunks + j}" for j in range(len(batch_chunks))],
                        )
                        total_chunks += len(batch_chunks)

                    except Exception as e:
                        logger.error(f"Ошибка при добавлении в ChromaDB: {e}")

                    # Очистка памяти
                    del batch_chunks, batch_metadatas, batch_embeddings

            except Exception as e:
                logger.error(f"Ошибка при обработке документа: {e}")

        # Логирование итогового потребления памяти и количества чанков
        logger.info(
            f"Итоговое потребление памяти: "
            f"{psutil.Process().memory_info().rss / 1024**2:.2f} МБ"
        )
        logger.info(f"✅ Загружено в коллекцию {total_chunks} чанков.")