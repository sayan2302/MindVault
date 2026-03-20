from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from app.core.config import settings


def get_chat_model() -> ChatGroq:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Set it in .env")
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def generate_answer(prompt: str) -> str:
    llm = get_chat_model()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)