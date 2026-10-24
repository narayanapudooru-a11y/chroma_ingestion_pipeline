from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import streamlit as st
from chromadb.config import Settings
from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

load_dotenv()

import ingest


@dataclass
class DocumentRecord:
    source_path: str | None
    source_name: str
    file_type: str
    file_size: int | None
    file_mtime: float | None
    chunk_count: int = 0
    ids: list[str] = field(default_factory=list)


@st.cache_resource
def get_client() -> chromadb.ClientAPI:
    ingest.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(ingest.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


@st.cache_resource
def get_collection() -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=ingest.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "description": "Incremental local PDF/DOCX ingestion"},
    )


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(ingest.EMBEDDING_MODEL)


def list_documents(collection: chromadb.Collection) -> list[DocumentRecord]:
    records: dict[str, DocumentRecord] = {}
    offset = 0
    page_size = 1000

    while True:
        result = collection.get(
            limit=page_size,
            offset=offset,
            include=["metadatas"],
        )
        ids = result.get("ids", [])
        if not ids:
            break

        metadatas = result.get("metadatas") or []
        for document_id, metadata in zip(ids, metadatas):
            metadata = metadata or {}
            source_path = metadata.get("source_path")
            if not isinstance(source_path, str) or not source_path:
                source_path = None
                key = f"chunk:{document_id}"
            else:
                key = source_path

            record = records.get(key)
            if record is None:
                source_name = metadata.get("source_name")
                file_type = metadata.get("file_type")
                file_size = metadata.get("file_size")
                file_mtime = metadata.get("file_mtime")
                record = DocumentRecord(
                    source_path=source_path,
                    source_name=source_name if isinstance(source_name, str) else (
                        Path(source_path).name if source_path else str(document_id)
                    ),
                    file_type=file_type if isinstance(file_type, str) else "",
                    file_size=file_size if isinstance(file_size, int) else None,
                    file_mtime=file_mtime if isinstance(file_mtime, (int, float)) else None,
                )
                records[key] = record
            record.ids.append(document_id)
            record.chunk_count += 1

        offset += len(ids)
        if len(ids) < page_size:
            break

    return sorted(records.values(), key=lambda record: (record.source_name.lower(), record.source_path or ""))


def _safe_source_file(relative_path: str) -> Path:
    root = ingest.SOURCE_DIR.resolve()
    source_file = root / relative_path
    resolved_file = source_file.resolve()
    try:
        resolved_file.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete a path outside the source directory: {relative_path}") from exc
    return source_file


def delete_document(collection: chromadb.Collection, record: DocumentRecord) -> int:
    if record.source_path:
        result = collection.get(where={"source_path": record.source_path}, include=[])
        ids = result.get("ids", [])
        source_file = _safe_source_file(record.source_path)
        if source_file.is_file() or source_file.is_symlink():
            source_file.unlink()
    else:
        ids = record.ids

    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _new_upload_path(filename: str) -> Path:
    basename = Path(filename).name
    extension = Path(basename).suffix.lower()
    stem = Path(basename).stem
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    if not safe_stem:
        safe_stem = "uploaded_document"

    ingest.SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = ingest.SOURCE_DIR / f"{safe_stem}{extension}"
    suffix = 2
    while candidate.exists():
        candidate = ingest.SOURCE_DIR / f"{safe_stem} ({suffix}){extension}"
        suffix += 1
    return candidate


def index_upload(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> int:
    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in ingest.SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and DOCX files are supported.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temporary_path = Path(temporary_file.name)

        extracted_text = ingest.extract_text(temporary_path)
        if not ingest.chunk_text(extracted_text, ingest.CHUNK_SIZE_CHARS, ingest.CHUNK_OVERLAP_CHARS):
            raise ValueError(
                f"No extractable text in {uploaded_file.name}. It may be scanned/image-only; run OCR first."
            )

        target = _new_upload_path(uploaded_file.name)
        shutil.copyfile(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    stat = target.stat()
    source = ingest.SourceFile(
        path=target,
        relative_path=target.relative_to(ingest.SOURCE_DIR).as_posix(),
        sha256=ingest.sha256_file(target),
        mtime=stat.st_mtime,
        size=stat.st_size,
    )
    return ingest.upsert_document(collection, model, source)


def _format_size(size: int | None) -> str:
    if size is None:
        return "—"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def main() -> None:
    st.set_page_config(page_title="Document Library", page_icon="📚", layout="wide")
    st.title("Document Library")
    st.caption("Upload and manage PDF and DOCX documents indexed in ChromaDB.")

    collection = get_collection()

    with st.container(border=True):
        st.subheader("Upload documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        if st.button(
            "Upload and index",
            type="primary",
            disabled=not uploaded_files,
        ):
            with st.spinner("Indexing uploaded documents..."):
                model = get_embedding_model()
                indexed = 0
                for uploaded_file in uploaded_files or []:
                    try:
                        chunks = index_upload(uploaded_file, collection, model)
                        indexed += 1
                        st.success(f"Indexed {uploaded_file.name} ({chunks} chunks).")
                    except Exception as exc:
                        st.error(f"Could not index {uploaded_file.name}: {exc}")

    records = list_documents(collection)
    total_chunks = sum(record.chunk_count for record in records)
    st.subheader(f"Indexed documents ({len(records)})")
    st.caption(f"{total_chunks} total chunks")

    if not records:
        st.info("No indexed documents yet. Upload a PDF or DOCX file to get started.")
        return

    header = st.columns([5, 1, 1, 1, 1])
    header[0].markdown("**Document**")
    header[1].markdown("**Type**")
    header[2].markdown("**Size**")
    header[3].markdown("**Chunks**")
    header[4].markdown("**Action**")

    for record in records:
        columns = st.columns([5, 1, 1, 1, 1], vertical_alignment="center")
        with columns[0]:
            st.write(record.source_name)
            st.caption(record.source_path or "No source path metadata")
        columns[1].write(record.file_type.upper() or "—")
        columns[2].write(_format_size(record.file_size))
        columns[3].write(record.chunk_count)
        key = record.source_path or (record.ids[0] if record.ids else record.source_name)
        button_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if columns[4].button("Delete", key=f"delete-{button_key}", type="secondary"):
            try:
                deleted = delete_document(collection, record)
            except Exception as exc:
                st.error(f"Could not delete {record.source_name}: {exc}")
            else:
                st.success(f"Deleted {record.source_name} ({deleted} chunks).")
                st.rerun()


if __name__ == "__main__":
    main()
