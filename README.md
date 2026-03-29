# AutoRAG

Portable tool that converts documents into RAG-ready chunked output folders. Supports PDF, EPUB, DOCX, TXT, MD, MOBI. Scanned pages are automatically OCR'd.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

### Desktop GUI

Double-click `AutoRAG.pyw` or:

```bash
autorag gui
```

Select one or more files, pick settings, click Process. Output goes to `./output/{filename}/` with chunks, markdown, and metadata.

### CLI

```bash
# Process a single file
autorag process document.pdf

# Process multiple files
autorag process file1.pdf file2.docx file3.epub -o ./output

# Process an entire directory of documents
autorag process ./my-documents/ -o ./rag-output

# Custom chunk size and strategy
autorag process document.pdf --max-tokens 1024 --strategy hierarchical

# JSON output for piping to other tools (stdout)
autorag process document.pdf --json

# Quiet mode (no progress bars)
autorag process document.pdf -q --json

# File info without processing
autorag info document.pdf
```

### Python API

```python
from pathlib import Path
from autorag.config import ProcessingConfig
from autorag.pipeline import process_document
from autorag.writer import write_output

result = process_document(Path("document.pdf"), ProcessingConfig())
doc_dir = write_output(result, Path("output"))

# Access results directly
for chunk in result.chunks:
    print(chunk.chunk_id, chunk.token_count, chunk.text[:80])
```

### n8n Integration

Use an **Execute Command** node:

- **Command:** `autorag process /path/to/file.pdf -o /path/to/output -q --json`
- **Output:** One JSON object per file on stdout:

```json
{"file": "document.pdf", "output": "output/document", "pages": 25, "chunks": 27, "ocr_used": false, "time_seconds": 1.4}
```

Pipe the JSON output to subsequent nodes for embedding, vector DB ingestion, etc.

## Output Structure

Each processed document produces a subfolder:

```
output/{filename}/
  manifest.json            # Run metadata, settings, stats
  chunks/chunk_001.json    # Individual chunk files
  chunks_combined.jsonl    # All chunks, one per line
  markdown/full_document.md
  config_snapshot.toml     # Settings used for this run
```

## Supported Formats

| Format | Extensions |
|--------|-----------|
| PDF | `.pdf` |
| EPUB | `.epub` |
| Word | `.docx` |
| Text | `.txt`, `.md`, `.rst` |
| MOBI | `.mobi` |
| XPS | `.xps`, `.oxps` |
| FictionBook | `.fb2` |
| Comic Book | `.cbz` |

## Settings

| Setting | Default | Options |
|---------|---------|---------|
| Chunk size | 512 tokens | 128, 256, 512, 1024, 2048 |
| Strategy | hybrid | hybrid, hierarchical |
| Overlap | auto (~12%) | Calculated from chunk size |
| OCR | auto | Triggers only on scanned pages |
