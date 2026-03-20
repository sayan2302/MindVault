from langchain_core.documents import Document


SYSTEM_RULES = (
    "You are MindVault, a grounded assistant. "
    "Answer ONLY from the provided context. "
    "If the answer is not in context, say: 'I could not find that in the provided documents.' "
    "Be concise and factual."
)


def build_context_block(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "na")
        blocks.append(
            f"[Source {i} | file={source} | chunk={chunk_index}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(user_query: str, docs: list[Document]) -> str:
    context_block = build_context_block(docs)
    return (
        f"{SYSTEM_RULES}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {user_query}\n\n"
        "Instructions:\n"
        "1) Use only context above.\n"
        "2) If insufficient context, say so clearly.\n"
        "3) End with a short 'Sources used' line mentioning source numbers."
    )