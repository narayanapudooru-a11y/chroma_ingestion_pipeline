# Architecture

## Overview
This project implements a local document ingestion pipeline that scans a source directory for PDF and DOCX files, extracts text, splits it into chunks, embeds the chunks with a Sentence Transformers model, and stores the vectors in a persistent ChromaDB collection for later semantic search and retrieval.

The solution includes two main runtime paths:
- A scheduled batch ingestion script: `ingest.py`
- A Streamlit document management UI: `app.py`

Both paths share the same ingestion logic and persistent Chroma configuration so the library view and the scheduled pipeline stay consistent.

## High-level component view

```text
+-----------------------------------------------------------+
| Source document directory (INGEST_SOURCE_DIR)             |
| - .pdf, .docx files                                       |
| - recursive scan                                           |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
| Ingestion orchestrator (`ingest.py`)                       |
| - discover_files()                                         |
| - compute SHA-256 hashes                                   |
| - compare against Chroma metadata                          |
| - add/update/delete reconciliation                          |
+-----------------------------+-----------------------------+
                              |
        +---------------------+---------------------+
        |                                           |
        v                                           v
+---------------------------+     +-------------------------------+
| Extractors                |     | Chunking / deterministic IDs  |
| PDF -> text              |     | - chunk_text()                |
| DOCX -> paragraphs/tables|     | - stable_chunk_id()           |
+---------------------------+     +-------------------------------+
        |                                           |
        v                                           |
        +-------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
| Embedding layer                                            |
| SentenceTransformer(EMBEDDING_MODEL)                       |
| normalize_embeddings=True                                  |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
| ChromaDB persistent collection                             |
| - collection name: CHROMA_COLLECTION                      |
| - storage: CHROMA_PERSIST_DIR                             |
| - cosine index (hnsw:space=cosine)                        |
| - ids + documents + metadata                               |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
| Operational outputs                                        |
| - stdout logs                                             |
| - logs/ingestion.log                                       |
| - partial-failure exit codes                              |
+-----------------------------------------------------------+
```

## Runtime responsibilities

### 1. Source discovery and reconciliation
The ingestion process starts by recursively scanning the configured source directory and collecting all supported document files.

For each file, it records:
- relative path inside the source root
- file hash (SHA-256)
- modification time
- file size
- file extension

The script compares this live inventory with the metadata already stored in Chroma. This lets it classify files as:
- new
- changed
- unchanged
- deleted from source tree

This reconciliation is important because the collection is persistent and must be updated incrementally instead of re-indexing everything on every run.

### 2. Text extraction
Each supported document type is handled by a dedicated extractor:
- `extract_pdf()` uses `pypdf.PdfReader` and reads page-by-page text
- `extract_docx()` uses `python-docx` and reads paragraphs plus table cell text

The extraction layer intentionally preserves page boundaries and table text where possible so the resulting chunks are more useful for retrieval.

### 3. Chunking
After text extraction, `chunk_text()` splits the content into character-based chunks using the configured:
- `CHUNK_SIZE_CHARS`
- `CHUNK_OVERLAP_CHARS`

The chunking logic is designed to:
- avoid empty text from scanned-only documents
- prefer natural boundaries such as newline or sentence splits when possible
- keep deterministic chunk IDs stable across runs

Each chunk gets a stable ID derived from:
- source path
- content hash
- chunk index

That makes reprocessing deterministic and helps prevent duplicate or drifted vectors when a file is updated.

### 4. Embedding and upsert
For each changed/new document, the script:
- builds a list of chunk texts
- computes embeddings using `SentenceTransformer`
- upserts the vectors into the collection with a document payload and metadata

Important behavior:
- Embedding is batched (`EMBED_BATCH_SIZE`)
- retries are applied to transient failures (`MAX_RETRIES`, `RETRY_DELAY_SECONDS`)
- if an embedding or upsert fails, the process logs the problem and continues with other files
- old document chunks are removed after the new version is successfully written

This ensures the collection is replaced atomically at the document level while preserving previous vectors if a new version cannot be ingested successfully.

### 5. Deletion logic
The pipeline also removes vectors for files that are no longer present in the source folder.

The delete flow:
- scans current source paths
- compares with metadata stored in Chroma
- deletes all vectors where `source_path` no longer exists in the source directory

This keeps the collection aligned with the authoritative document store.

## Data model and storage

### Chroma collection metadata
Each vector record includes metadata such as:
- `source_path`: relative path in the source directory
- `source_name`: original filename
- `file_hash`: SHA-256 digest for change detection
- `chunk_index`: chunk position in document
- `file_mtime`: modification timestamp
- `file_size`: file size
- `file_type`: extension without the leading dot

The collection is created with:
- name: `CHROMA_COLLECTION`
- metadata: `{ "hnsw:space": "cosine", "description": "Incremental local PDF/DOCX ingestion" }`

This uses cosine similarity for vector comparison and retrieval.

### Persistent local storage
Chroma operates in a persistent local directory defined by `CHROMA_PERSIST_DIR`.

This is the stateful part of the system and should be treated as mission-critical because it contains the vector database, embedding index, and metadata that drive retrieval.

## Streamlit document library
The `app.py` module provides a browser-based management interface for the same document corpus.

It is organized around these functions:
- `get_client()`
- `get_collection()`
- `get_embedding_model()`
- `list_documents()`
- `index_upload()`
- `delete_document()`
- `main()`

The UI allows users to:
- upload one or more PDF/DOCX files
- index them into the configured collection
- list indexed documents with metadata
- delete a document and its source file plus its Chroma vectors

This app shares the same ingestion primitives as the CLI script so it behaves consistently for local management workflows.

## Control flow

### Batch ingestion
```text
run_ingestion()
  -> discover_files()
  -> create or open PersistentClient
  -> get_or_create_collection()
  -> load existing Chroma metadata
  -> for each file:
       if unchanged -> skip
       else -> upsert_document()
       -> embed and store chunks
       -> remove stale chunk versions
  -> for each missing source path:
       -> delete vectors from Chroma
  -> log summary and exit non-zero on failures
```

### Upload/index flow in Streamlit
```text
main()
  -> get_collection()
  -> user uploads file(s)
  -> index_upload()
       -> validate extension
       -> write temporary file
       -> extract text
       -> validate chunk generation
       -> save to source directory
       -> create SourceFile
       -> upsert_document()
  -> display success/error feedback
```

## Configuration surface
The project relies on environment variables and defaults defined in `ingest.py`:

- `INGEST_SOURCE_DIR` — source document directory
- `CHROMA_PERSIST_DIR` — Chroma persistent storage path
- `CHROMA_COLLECTION` — collection name
- `EMBEDDING_MODEL` — Sentence Transformers model
- `CHUNK_SIZE_CHARS` — approximate chunk size
- `CHUNK_OVERLAP_CHARS` — chunk overlap
- `EMBED_BATCH_SIZE` — per-request embedding batch size
- `MAX_RETRIES` / `RETRY_DELAY_SECONDS` — transient failure handling
- `LOG_LEVEL` / `INGEST_LOG_DIR` — logging configuration

The app loads `.env` automatically via `python-dotenv`, allowing local development configuration without permanent shell state.

## Operational notes

### Why this architecture works well for local deployments
- It is simple and deterministic.
- It avoids re-embedding unchanged files.
- It persists the vector store locally so it survives process restarts.
- It is suitable for scheduled, low-maintenance runs in a single-machine environment.

### Important constraints
- Only text-based PDFs and DOCX files are supported directly.
- Scanned/image-only PDFs require OCR before ingestion.
- Multiple writers should not share the same local Chroma directory concurrently.
- The source directory and Chroma persistence directory should be kept on reliable local storage.
- For high-scale or multi-process use, a dedicated Chroma server deployment is preferred.

## Summary
This architecture is a lightweight, local-first retrieval pipeline built around:
- filesystem-based document ingestion
- deterministic incremental indexing
- sentence-transformer embeddings
- persistent Chroma vector storage
- an optional Streamlit management interface

It is designed for a single-user or small-team environment where local document ingestion, semantic indexing, and repeatable scheduled runs are more important than distributed deployment complexity.
