from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import ensure_runtime_dirs

app = FastAPI(title="MindVault API", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    ensure_runtime_dirs()


app.include_router(api_router)