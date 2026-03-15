from fastapi import APIRouter, HTTPException, status, UploadFile, File
from uuid import uuid4
from pathlib import Path
from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

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