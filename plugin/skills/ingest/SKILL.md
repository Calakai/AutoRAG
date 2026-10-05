---
name: ingest
description: Add files or folders to the user's local AutoRAG document library. Only when the user explicitly asks to index, ingest or add documents.
argument-hint: <paths...> [into <collection>] [tags a,b]
disable-model-invocation: true
allowed-tools: mcp__plugin_autorag_autorag__ingest_documents, mcp__plugin_autorag_autorag__list_collections
---

Add documents to the AutoRAG library. Request: $ARGUMENTS

1. Work out the paths, the collection (default `default`) and any tags from the request.
   Call `list_collections` to reuse an existing collection name when the user's wording
   matches one.
2. If no paths were given, ask for them. Do not guess paths.
3. Before ingesting, state in one line what you will add and where, e.g.
   "Adding ~/Documents/manuals (recursive) to collection `manuals`, tags: hardware".
4. Call `ingest_documents` with absolute paths. Use `recursive: true` for folders unless the
   user said otherwise.
5. Report the outcome as a short table: added / updated / unchanged / failed counts, then
   each failure with its reason, and any "scanned pages unread" notes (these need the
   `[ocr]` extra: `pip install "autorag[ocr]"`).

If the tool is unavailable, AutoRAG is not installed: tell the user to run
`pip install "autorag[all] @ git+https://github.com/Calakai/AutoRAG"` and restart Claude Code.
