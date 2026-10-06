# AutoRAG — Agent Guide

> Canonical instruction file, read by Codex, Cursor and Claude Code (via `CLAUDE.md` →
> `@AGENTS.md`). It holds what the code can't tell you. Anything findable with `ls`,
> `pyproject.toml` or a docstring is deliberately absent.

## What this is

A local document ingestion and search service that sits **beside** Cal's apps and agents
rather than inside them: files → chunks with provenance → local embeddings → one SQLite
file, exposed over a CLI, a Python API (`autorag.kb.KnowledgeBase`) and an MCP server.

MonkChat and Aegis each own their RAG stack. AutoRAG was ported into MonkChat
(`importing.py`, `rag.py`) and removed from Aegis (`CHANGELOG.md:441`) on purpose. Do not
propose making AutoRAG a runtime dependency of either app. Integrate through files, MCP, or
the shared embedding recipes below.

## Rules that bite

- 🔴 **Core dependencies stay permissively licensed.** PyMuPDF (AGPL) was removed because it
  blocked MIT and MonkChat. Never reintroduce it or any copyleft package, not even as an
  extra. CI's `core-install` job fails if a core install pulls an optional or copyleft
  package.
- 🔴 **The core path never downloads a model.** Extracting and chunking a file must work
  offline on first run. That is why Docling's HybridChunker (HuggingFace tokenizer fetch) was
  replaced by `chunking.py`. Model downloads belong only to the optional embedders.
- 🔴 **Tests never touch the network.** Use `tests.helpers.HashingEmbedder` and the
  `make_pdf`/`make_epub`/`make_fb2`/`make_xps` builders. CI sandboxes and Claude's cloud
  sessions cannot reach HuggingFace.
- **A collection is pinned to its embedder** (`collections.embedder`). Mixing vectors from
  two models gives wrong results with no error, so a mismatch raises
  `EmbedderMismatchError`. Changing models means a new collection or a re-index, never a
  silent fallback.
- **`aegis` embedder = Aegis's exact recipe:** Ollama `nomic-embed-text`, `search_document: `
  / `search_query: ` prefixes, sliced to 512 dims and re-normalised. It must match
  `aegis/src/lib/ai/provider.ts` (`toUnit512`). If Aegis changes its embedder, change this
  in the same breath.
- **Embedders raise `EmbeddingUnavailableError` and never return empty.** "No embedder" must
  never look like "no results".
- **Retrieved text is untrusted data.** MCP output wraps it in `<excerpt>` and the server
  instructions tell the model not to follow directions found inside documents. Keep both.
- **Paged-format text cleaning (`extract.clean_text`) is for PDF/ebook/OCR output only.**
  Applied to markdown, CSV-like or docx text, it deletes legitimate short lines and numbers.
- **`max_tokens` is an estimate** (`chunking.estimate_tokens`, the larger of chars/4 and
  words×1.3), conservative for BERT-style tokenizers. The default embedder
  (`bge-small-en-v1.5`) truncates input at 512 tokens.

- 🔴 **Plugin skills must name plugin agents by namespace** (`agent: autorag:librarian`). A
  bare `agent: librarian` is not an error: the skill forks with the *full* tool set, so the
  librarian could ingest and run Bash. Verified in a live session on 2026-10-05.
  `tests/test_plugin.py` guards it, and also checks that the librarian's tool list stays
  read-only.

- **Postgres goes through pg8000 (BSD).** psycopg 2 and 3 are LGPL, so they're ruled out
  by the no-copyleft rule.
- **`export_aegis` mirrors `aegis/src/app/api/uploads/route.ts`:** book naming
  (`deriveTitleFromName`), the reserved-name regex, the category hints, delete-then-insert
  per book, and the `user_library_books` upsert. If that route changes, change this too.
  `tests/test_export.py` runs Aegis's real `match_user_library` against exported rows.
- **SQL sources are read-only, and DSNs are never stored.** Row identity is
  `sql:<label>/<id>`. Only `redact_dsn` output may be printed.

## Conventions

- `pip install -e ".[dev]"` then `pytest -q`. Every new module ships tests, and a PR never
  lowers the test count.
- Supported Python: 3.10–3.13 (CI matrix). Both MCP Python SDK lines are supported: 2.x
  `MCPServer` and 1.x `FastMCP` from 1.14 (earlier 1.x releases choke on postponed annotations).
- Optional features are extras (`embed`, `ocr`, `mcp`, `gui`) with an import check and a
  friendly install hint. A missing extra is never a traceback.
- Data lives in `$AUTORAG_HOME` (default `~/.autorag/knowledge.db`).

## Roadmap (agreed 2026-10-05)

1. ✅ Foundation: MIT-clean extraction, offline chunker, local embeddings, SQLite hybrid
   search, MCP server.
2. ✅ Claude Code plugin (`plugin/`): `/autorag:ingest`, `/autorag:ask-docs` and a read-only
   librarian agent. The marketplace lives at the repo root.
3. ✅ SQL tables as a source (`index-sql`); exports to Postgres/pgvector (`autorag_chunks`)
   and to Aegis's player library (`export --to aegis`).
4. **Prove value before anything else** (Caleb, 2026-10-06). AutoRAG has not yet been
   run on real models or a real corpus, and nobody has measured its retrieval quality.
   Next step: index a real document set, ask about 10 real questions, and compare
   `/autorag:ask-docs` against plain Claude Code reading the same files. If it doesn't
   clearly win, archive it.
   - 🔴 **No integration into MonkChat, Aegis or other apps until then.** Don't add it as a
     dependency, and don't plan app work around it. The existing `export --to aegis` stays
     as-is but isn't a roadmap commitment: it targets Aegis's Postgres layer, which the
     Swift rebuild is replacing.
