from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings


def get_embedding_client() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def embed_documents(documents: list[Document]) -> list[list[float]]:
    embedding_client = get_embedding_client()
    texts = [doc.page_content for doc in documents]
    return embedding_client.embed_documents(texts)


def embed_query_text(query: str) -> list[float]:
    embedding_client = get_embedding_client()
    return embedding_client.embed_query(query)