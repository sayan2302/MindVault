from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

def load_documents_from_file(file_path: Path) -> list[Document]:

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == ".md":
        loader = UnstructuredMarkdownLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    
    docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata["source_file"] = file_path.name
        doc.metadata["file_type"] = suffix
        doc.metadata["doc_index"] = i
    
    return docs