# Incremental ChromaDB ingestion

## What it does
- Scans a configured local folder recursively for `.pdf` and `.docx`.
- Uses SHA-256 content hashes, so renames, additions, edits, and deletions are detected.
- Embeds only new/changed documents; unchanged documents are skipped.
- Writes vectors to a persistent Chroma collection.
- Removes vectors for files that have disappeared from the scanned folder.
- Logs to stdout and `logs/ingestion.log`; exits non-zero if any file fails.

## Important notes
- This supports text-based PDFs and DOCX. Scanned/image-only PDFs need OCR first.
- Legacy `.doc` files are not supported directly. Convert them to `.docx` (for example with LibreOffice) before ingestion.
- Keep the Chroma data directory on a reliable local disk. Do not let multiple independent processes write to the same local Chroma directory concurrently.
- Back up the Chroma persistence directory before upgrades or bulk changes.
- For large collections or multiple application instances, use a supported Chroma server deployment rather than sharing a local persistence directory over a network drive.
- A content hash is computed each run, which reads every supported file but does not re-embed unchanged files.
- The script uses character-based chunking for portability. For complex layouts, tables, or scanned PDFs, consider a document parser/OCR pipeline.

## Install
Use Python 3.11 or 3.12 in a virtual environment (recommended):
```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run may download the Sentence Transformers model from Hugging Face. In restricted production networks, pre-download/cache the model and validate outbound access requirements.

## Streamlit document library
The Streamlit app uploads and indexes PDF and DOCX documents in the configured source folder, and lists indexed documents with a delete action. Deleting a document removes its source file and Chroma vectors so a later ingestion run will not restore it.

Configure the same `INGEST_SOURCE_DIR`, `CHROMA_PERSIST_DIR`, and `CHROMA_COLLECTION` settings used by the ingestion pipeline in `.env` or the process environment, then start the app from the project directory:
```powershell
streamlit run .\\app.py
```

The app loads `.env` automatically. The first upload may download the Sentence Transformers embedding model.

## Configure on Windows
Set environment variables before running, or set them in the Task Scheduler action. Example PowerShell:
```powershell
$env:INGEST_SOURCE_DIR = "D:\\knowledge_documents"
$env:CHROMA_PERSIST_DIR = "D:\\chroma_data"
$env:INGEST_LOG_DIR = "D:\\chroma_ingestion_logs"
python .\\ingest.py
```

Set paths to real directories. The source directory must exist; the Chroma directory can be created automatically.

## Schedule every Saturday at 08:00 IST on Windows
Use **Task Scheduler**:
1. Create Task (not Basic Task), name it `Chroma weekly ingestion`.
2. Trigger: Weekly, Saturday, start time `8:00:00 AM`. Ensure Windows time zone is set to India Standard Time.
3. Action: Start a program.
   - Program: full path to `.venv\\Scripts\\python.exe`
   - Arguments: full path to `ingest.py`
   - Start in: project directory
4. Set “If the task is already running” to **Do not start a new instance**.
5. Configure the task to run whether user is logged on or not, and ensure the task account can read source files and write the Chroma/log directories.
6. Enable task history. Configure monitoring/alerting for non-zero exit codes and log errors.

You can also test it manually before scheduling:
```powershell
python .\\ingest.py
```

## Operational recommendations
- Run the first full ingestion in a maintenance window; embeddings are CPU/RAM intensive.
- Test add/edit/delete behavior in a non-production folder and collection first.
- Back up Chroma before first production use and before dependency upgrades.
- Do not delete or rename the source directory during a run.
- For strict consistency, run one ingestion process at a time and monitor non-zero exit status.
