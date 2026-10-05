# AutoRAG

Local document ingestion and search for your apps and AI agents.

- **Ingest:** PDF, EPUB, FB2, XPS/OXPS, DOCX, HTML, TXT, MD and RST. Scanned pages are OCR'd when the `[ocr]` extra is installed.
- **Chunk:** splits by document structure (headings → sections), with page and heading provenance.
- **Embed locally:** fastembed (ONNX on CPU) or Ollama. Nothing leaves your machine.
- **Search:** hybrid semantic + keyword search over a single SQLite file.
- **Use it from anywhere:** an MCP server for Claude, a CLI with JSON output for n8n and scripts, and a Python API.

MIT licensed, with no copyleft dependencies.

## Install

```bash
pip install -e ".[all]"   # embeddings + OCR + MCP server
pip install -e .           # core only: extract + chunk to folders, no models
```

| Extra | Adds |
|---|---|
| `embed` | fastembed for local embeddings (`BAAI/bge-small-en-v1.5`, downloaded once) |
| `ocr` | OCR for scanned pages: Apple Vision on macOS, RapidOCR elsewhere |
| `mcp` | The MCP server |
| `postgres` | Postgres as a SQL source and pgvector export target (pg8000) |
| `gui` | The original desktop app (CustomTkinter) |

## Knowledge base

```bash
# Add a folder (unchanged files are skipped on re-runs)
autorag index ~/Documents/manuals -c manuals -r --tags hardware

# Search: hybrid by default, citations included
autorag search "how do I reset the router" -c manuals
autorag search "reset router" --mode keyword --json

# Manage
autorag collections
autorag docs -c manuals
autorag rm "router-guide.pdf" -c manuals
```

- **Storage:** everything lives in `~/.autorag/knowledge.db`. Override it with `--db` or `AUTORAG_HOME`.
- **Collections:** group documents, and each is pinned to the embedder that built it.

### Embedders

| `--embedder` | What it does |
|---|---|
| `fastembed` (default) | `BAAI/bge-small-en-v1.5`, 384 dims, CPU |
| `fastembed:<model>` | Any fastembed model |
| `ollama:<model>[@dims]` | A running Ollama server (`AUTORAG_OLLAMA_URL`, default `http://localhost:11434`) |
| `aegis` | Aegis's exact recipe: `nomic-embed-text`, its prefixes, 512 dims. Vectors are interchangeable with Aegis's pgvector columns |

Set a default with `AUTORAG_EMBEDDER`.

## SQL tables as a source

Index rows from SQLite or Postgres. Each row becomes one document (`sql:<label>/<id>`).

```bash
export CRM_DSN="postgresql://cal:…@localhost:5432/crm"        # keeps the password out of history
autorag index-sql --dsn env:CRM_DSN --label crm \
  --query "SELECT id, subject, body, status FROM tickets" \
  --id-column id --title-column subject --text-columns body --meta-columns status \
  -c support --sync
```

- **Read-only:** SQLite opens in `mode=ro`, and Postgres sessions run `READ ONLY`, so the query can't change anything.
- **Incremental:** unchanged rows are skipped on re-runs. `--sync` removes documents whose rows are gone, and only touches documents from that `--label`.
- **Postgres** needs the `[postgres]` extra (pg8000, which is BSD licensed).

## Export to other databases

The local library stays the source of truth. An export **replaces** that collection's rows in the target, so re-run it to refresh.

```bash
# Any Postgres with pgvector (local, Docker, Supabase)
autorag export -c support --to env:AUTORAG_EXPORT_DSN            # table autorag_chunks + HNSW index

# Aegis's player library
autorag index ~/dnd/books -r -c dnd --embedder aegis
AEGIS_SERVICE_KEY=<from `npm run db:jwt`> autorag export -c dnd --to aegis
```

| Target | What it writes |
|---|---|
| `postgres://…` | `autorag_chunks`, created if missing and owned by AutoRAG, with one row per chunk. One transaction, so the target never holds half an export. A table is fixed to one vector size; use `--table` for a different embedder |
| `aegis` | `user_library_chunks` + `user_library_books` through Aegis's PostgREST (`AEGIS_URL`, default `http://localhost:3001`). Same book names, reserved-name check, category guess and replace-by-book behaviour as Aegis's upload route. The collection must be built with `--embedder aegis` |

> **Aegis vectors:** same model, prefix and 512 dimensions as Aegis, so `match_user_library` finds them. AutoRAG embeds each chunk with its file and section names in front, while Aegis embeds the bare text, so vectors carry slightly more context than Aegis's own.

## MCP server (Claude Code, Claude Desktop, any MCP client)

```bash
claude mcp add autorag -- autorag mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "autorag": { "command": "autorag", "args": ["mcp"] } } }
```

**Tools:**

| Tool | Purpose |
|---|---|
| `search_knowledge` | Hybrid search, returns cited excerpts |
| `read_chunk` | A chunk plus its neighbours, for wider context |
| `list_collections` | Show the collections |
| `list_documents` | Show indexed documents |
| `ingest_documents` | Add files or folders |
| `remove_document` | Remove a document and its chunks |

**Safety:**
- **Allowed folders:** set `AUTORAG_ALLOWED_ROOTS` (folders separated by `:`, or `;` on Windows) to limit what the server may ingest.
- **What gets read:** ingestion only reads supported document formats and skips hidden files.

## Claude Code plugin

Adds cited document answers to every Claude Code session.

```bash
pip install "autorag[all] @ git+https://github.com/Calakai/AutoRAG"   # once; puts `autorag` on PATH
```
```
/plugin marketplace add calakai/autorag
/plugin install autorag@calakai
```

| Piece | Use |
|---|---|
| `/autorag:ask-docs <question>` | Runs the **librarian** agent and returns a cited answer. Claude can also use it on its own when a question looks like it is about your documents |
| `/autorag:ingest <paths> [into <collection>]` | Adds documents. Only runs when you type it |
| `librarian` agent | Read-only: search, read and list tools only. It cannot add or remove documents |

The plugin runs `autorag mcp`, so the `autorag` command must be on the `PATH` Claude Code starts with.

## Chunk folders (the original AutoRAG output)

```bash
autorag process document.pdf -o ./output
autorag process ./docs -r --max-tokens 1024 --strategy hierarchical --json
autorag info document.pdf
```

Each document gets its own folder:

```
output/{filename}/
  manifest.json            # run metadata, settings, stats
  chunks/chunk_001.json    # one file per chunk
  chunks_combined.jsonl    # all chunks, one per line
  markdown/full_document.md
  config_snapshot.toml
```

**n8n:** use an Execute Command node with `autorag process <file> -o <dir> -q --json` or `autorag search "<query>" --json`. Each prints one JSON object per line.

## Python API

```python
from autorag.kb import KnowledgeBase

with KnowledgeBase(embedder="fastembed") as kb:
    kb.ingest(["~/Documents/manuals"], collection="manuals", recursive=True)
    for hit in kb.search("reset the router", collection="manuals", k=5):
        print(hit.citation(), hit.score)
        print(hit.text)
```

## Settings

| Setting | Default | Options |
|---|---|---|
| Chunk size | 512 tokens (estimated) | Any value ≥ 16 |
| Strategy | `hybrid` | `hybrid` merges tiny sections and keeps their headings in the text. `hierarchical` makes one chunk per section |
| Overlap | 64 tokens (`max_tokens / 8` from the CLI) | Applied within a section only |
| Search | `hybrid` | `hybrid` (reciprocal-rank fusion), `vector`, `keyword` |

## Changes in 0.2

- **PyMuPDF (AGPL) replaced** by pypdfium2 plus stdlib ebook parsers. MOBI and CBZ are no longer supported.
- **Docling's chunker replaced** by an offline, structure-aware chunker. There is no HuggingFace download on first run.
- **Desktop GUI now optional**, via the `[gui]` extra.
- **Manifests changed:** `docling_version` becomes `chunker`, and `scanned_pages` / `ocr_pages_skipped` are new.
