from app.api.schemas import ChatResponse, SourceItem
from app.rag.llm import generate_answer
from app.rag.prompting import build_rag_prompt
from app.rag.vector_store import retrieve_similar_chunks


def run_rag_pipeline(query: str, top_k: int) -> ChatResponse:
    docs = retrieve_similar_chunks(query=query, k=top_k)

    prompt = build_rag_prompt(user_query=query, docs=docs)
    answer = generate_answer(prompt)

    sources: list[SourceItem] = []
    for doc in docs:
        sources.append(
            SourceItem(
                source_file=doc.metadata.get("source_file"),
                chunk_index=doc.metadata.get("chunk_index"),
                preview=doc.page_content[:180],
            )
        )

    return ChatResponse(
        query=query,
        top_k=top_k,
        answer=answer,
        sources=sources,
    )