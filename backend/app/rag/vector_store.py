from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.core.config import settings
from uuid import uuid4


def get_embedding_client() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name="mindvault_docs",
        embedding_function=get_embedding_client(),
        persist_directory=settings.chroma_dir,
    )

def build_chunk_ids(chunks: list[Document]) -> list[str]:
    ids: list[str] = []
    for chunk in chunks:
        source = str(chunk.metadata.get("source_file", "unknown"))
        chunk_index = str(chunk.metadata.get("chunk_index", "na"))
        ids.append(f"{source}:{chunk_index}:{uuid4().hex[:8]}")
    return ids

def index_chunks(chunks: list[Document]) -> dict[str, int]:
    vector_store = get_vector_store()
    ids = build_chunk_ids(chunks)
    vector_store.add_documents(documents=chunks, ids=ids)
    return {"indexed_count": len(chunks)}

def retrieve_similar_chunks(query: str, k: int) -> list[Document]:
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query=query, k=k)
    return results


def retrieve_similar_chunks_with_scores(query: str, k: int) -> list[tuple[Document, float]]:
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query=query, k=k)
    return results