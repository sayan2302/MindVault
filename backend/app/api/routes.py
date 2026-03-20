from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from uuid import uuid4
from pathlib import Path
from app.core.config import settings
from app.rag.loaders import load_documents_from_file
from app.rag.chunking import chunk_documents
from app.rag.embeddings import embed_documents, embed_query_text
from app.rag.vector_store import index_chunks, retrieve_similar_chunks_with_scores

from app.rag.prompting import build_rag_prompt
from app.rag.llm import generate_answer
from app.rag.vector_store import retrieve_similar_chunks

router = APIRouter()

##################################################################################################

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


##################################################################################################
ALLOWED_EXTENSIONS = {".pdf",".txt",".md"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 # 10 MB

def _validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail = f"Unsupported file type: {extension}. Allowed: .pdf, .txt, .md"
        )
    return extension

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str,str|int] :
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing"
        )
    extension = _validate_extension(file.filename)
    content= await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 10 MB limit",
        )
    
    safe_stem = Path(file.filename).stem.replace(" ", "_")
    unique_name = f"{safe_stem}-{uuid4().hex[:8]}{extension}"
    destination = Path(settings.upload_dir) / unique_name
    destination.write_bytes(content)

    await file.close()

    return {
        "message": "Upload successful",
        "stored_filename": unique_name,
        "size_bytes": len(content),
        "storage_path": str(destination),
    }



##################################################################################################
@router.post("/injest/{stored_filename}")
def ingest_uploaded_file(stored_filename: str) -> dict[str, object]:
    file_path = Path(settings.upload_dir) / stored_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {stored_filename}",
        )
    
    try:
        docs = load_documents_from_file(file_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load document: {exc}",
        ) from exc
    
    sample = docs[0].metadata if docs else {}

    return {
        "message": "Ingestion load successful",
        "stored_filename": stored_filename,
        "documents_loaded": len(docs),
        "sample_metadata": sample,
    }



##################################################################################################
@router.post("/chunk/{stored_filename}")
def chunk_uploaded_file(stored_filename: str) -> dict[str, object]:
    file_path = Path(settings.upload_dir) / stored_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {stored_filename}",
        )

    try:
        docs = load_documents_from_file(file_path)
        chunks = chunk_documents(docs)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to chunk document: {exc}",
        ) from exc

    first_chunk_meta = chunks[0].metadata if chunks else {}
    first_chunk_preview = chunks[0].page_content[:200] if chunks else ""

    return {
        "message": "Chunking successful",
        "stored_filename": stored_filename,
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "first_chunk_metadata": first_chunk_meta,
        "first_chunk_preview": first_chunk_preview,
    }



##################################################################################################
@router.post("/embed/{stored_filename}")
def embed_uploaded_file(stored_filename: str) -> dict[str, object]:
    file_path = Path(settings.upload_dir) / stored_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {stored_filename}",
        )

    try:
        docs = load_documents_from_file(file_path)
        chunks = chunk_documents(docs)
        vectors = embed_documents(chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {exc}",
        ) from exc

    if not vectors:
        return {
            "message": "No vectors generated",
            "stored_filename": stored_filename,
            "chunks_count": 0,
        }

    return {
        "message": "Embedding successful",
        "stored_filename": stored_filename,
        "chunks_count": len(chunks),
        "vectors_count": len(vectors),
        "embedding_dimension": len(vectors[0]),
        "first_vector_preview": vectors[0][:8],
    }

@router.get("/embed-query")
def embed_query(query: str = Query(..., min_length=1)) -> dict[str, object]:
    try:
        vector = embed_query_text(query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query embedding failed: {exc}",
        ) from exc

    return {
        "query": query,
        "embedding_dimension": len(vector),
        "vector_preview": vector[:8],
    }

@router.post("/index/{stored_filename}")
def index_uploaded_file(stored_filename: str) -> dict[str, object]:
    file_path = Path(settings.upload_dir) / stored_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {stored_filename}",
        )

    try:
        docs = load_documents_from_file(file_path)
        chunks = chunk_documents(docs)
        result = index_chunks(chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {exc}",
        ) from exc

    return {
        "message": "Indexing successful",
        "stored_filename": stored_filename,
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "indexed_count": result["indexed_count"],
        "collection": "mindvault_docs",
        "persist_directory": settings.chroma_dir,
    }


@router.get("/retrieve")
def retrieve_chunks(
    query: str = Query(..., min_length=1),
    k: int = Query(default=settings.retrieval_top_k, ge=1, le=20),
) -> dict[str, object]:
    try:
        results = retrieve_similar_chunks_with_scores(query=query, k=k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    response_items: list[dict[str, object]] = []
    for rank, (doc, score) in enumerate(results, start=1):
        response_items.append(
            {
                "rank": rank,
                "score": score,
                "source_file": doc.metadata.get("source_file"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "preview": doc.page_content[:220],
                "metadata": doc.metadata,
            }
        )

    return {
        "query": query,
        "top_k": k,
        "results_count": len(response_items),
        "results": response_items,
    }



@router.get("/rag-answer")
def rag_answer(
    query: str = Query(..., min_length=1),
    k: int = Query(default=settings.retrieval_top_k, ge=1, le=20),
) -> dict[str, object]:
    try:
        docs = retrieve_similar_chunks(query=query, k=k)
        prompt = build_rag_prompt(user_query=query, docs=docs)
        answer = generate_answer(prompt)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG answer generation failed: {exc}",
        ) from exc

    sources: list[dict[str, object]] = []
    for i, doc in enumerate(docs, start=1):
        sources.append(
            {
                "source_no": i,
                "source_file": doc.metadata.get("source_file"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "preview": doc.page_content[:180],
            }
        )

    return {
        "query": query,
        "top_k": k,
        "answer": answer,
        "sources": sources,
    }