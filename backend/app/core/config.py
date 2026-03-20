from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: str = ""
    # groq_model: str = "llama-3.1-8b-instant"
    groq_model: str = "moonshotai/kimi-k2-instruct-0905"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "embeddinggemma"

    upload_dir: str = "data/uploads"
    chroma_dir: str = "chroma_db"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5


settings = Settings()


def ensure_runtime_dirs() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)