---
name: ask-docs
description: Answer a question from the user's local AutoRAG document library with citations. Use when the user asks about their own documents, manuals, books or notes, or says "check my docs", "what does the manual say", "according to my notes".
argument-hint: <question>
context: fork
agent: autorag:librarian
---

Answer this question from the user's document library: $ARGUMENTS

Return the answer with citations in the form `[file, p.X › Section]`. If the library does not
cover it, say so and list what you searched.
