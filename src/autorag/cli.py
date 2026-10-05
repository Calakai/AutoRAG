"""Command-line interface for AutoRAG."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from autorag import __version__


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _progress(message: str, fraction: float) -> None:
    bar_len = 30
    filled = int(bar_len * fraction)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stderr.write(f"\r  [{bar}] {fraction:3.0%} {message[:60]:<60}")
    sys.stderr.flush()
    if fraction >= 1.0:
        sys.stderr.write("\n")


def _kb(args: argparse.Namespace):
    from autorag.kb import KnowledgeBase

    return KnowledgeBase(db_path=args.db, embedder=args.embedder)


# --- chunk folders (the original AutoRAG output) ---------------------------------------


def cmd_process(args: argparse.Namespace) -> None:
    from autorag.config import ChunkingConfig, GeneralConfig, ProcessingConfig
    from autorag.kb import collect_files
    from autorag.pipeline import process_document
    from autorag.writer import write_output

    files = collect_files(args.files, recursive=args.recursive)
    if not files:
        print("Error: no files to process", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    config = ProcessingConfig(
        general=GeneralConfig(output_dir=str(output_dir)),
        chunking=ChunkingConfig(
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            overlap_tokens=max(0, args.max_tokens // 8),
        ),
    )

    total_chunks = 0
    failed = []
    start = time.monotonic()
    for i, file_path in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {file_path.name}", file=sys.stderr)
        try:
            callback = _progress if not args.quiet else None
            result = process_document(file_path, config, progress_callback=callback)
            doc_dir = write_output(result, output_dir)
            total_chunks += result.total_chunks
            print(f"  -> {result.total_chunks} chunks in {result.processing_time_seconds}s", file=sys.stderr)
            if result.ocr_pages_skipped:
                print(
                    f"  !! {result.ocr_pages_skipped} scanned page(s) unread — install the [ocr] extra",
                    file=sys.stderr,
                )
            if args.json:
                print(json.dumps({
                    "file": str(file_path),
                    "output": str(doc_dir),
                    "pages": result.total_pages,
                    "chunks": result.total_chunks,
                    "ocr_used": result.ocr_used,
                    "time_seconds": result.processing_time_seconds,
                }))
        except Exception as e:
            failed.append(str(file_path))
            print(f"  ERROR: {e}", file=sys.stderr)

    elapsed = time.monotonic() - start
    print(f"\nDone: {total_chunks} chunks from {len(files)} files in {elapsed:.1f}s", file=sys.stderr)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Show info about a file without processing it."""
    from autorag.extract import SUPPORTED_EXTENSIONS, page_count

    path = Path(args.file)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    ext = path.suffix.lower()
    info: dict = {"file": str(path), "format": ext, "size_bytes": path.stat().st_size,
                  "supported": ext in SUPPORTED_EXTENSIONS}
    if info["supported"]:
        pages = page_count(path)
        if pages is not None:
            info["pages"] = pages
    print(json.dumps(info, indent=2))


# --- knowledge base ---------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    def on_file(index: int, total: int, path: Path) -> None:
        if not args.quiet:
            print(f"[{index + 1}/{total}] {path.name}", file=sys.stderr)

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    with _kb(args) as kb:
        outcomes = kb.ingest(
            args.paths,
            collection=args.collection,
            tags=tags,
            recursive=args.recursive,
            force=args.force,
            on_file=on_file,
        )
    if not outcomes:
        print("Error: no supported documents found", file=sys.stderr)
        sys.exit(1)
    if args.json:
        for o in outcomes:
            print(json.dumps(asdict(o)))
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
        if o.status == "failed":
            print(f"  ERROR {Path(o.path).name}: {o.error}", file=sys.stderr)
        elif o.ocr_pages_skipped:
            print(f"  !! {Path(o.path).name}: {o.ocr_pages_skipped} scanned page(s) unread", file=sys.stderr)
    summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    print(f"Collection '{args.collection}': {summary}", file=sys.stderr)
    if counts.get("failed"):
        sys.exit(1)


def cmd_search(args: argparse.Namespace) -> None:
    with _kb(args) as kb:
        hits = kb.search(args.query, collection=args.collection, k=args.k, mode=args.mode)
    if args.json:
        for hit in hits:
            print(json.dumps(asdict(hit), ensure_ascii=False))
        return
    if not hits:
        print("No matches.", file=sys.stderr)
        return
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] {hit.citation()}  ({hit.chunk_id}, score {hit.score:.4f})")
        text = hit.text if args.full else (hit.text[:400] + ("…" if len(hit.text) > 400 else ""))
        print("    " + text.replace("\n", "\n    ") + "\n")


def cmd_collections(args: argparse.Namespace) -> None:
    with _kb(args) as kb:
        collections = kb.collections()
        active = kb.embedder.name
    if args.json:
        print(json.dumps([asdict(c) for c in collections], indent=2))
        return
    if not collections:
        print("No collections yet. Add documents with: autorag index <paths>", file=sys.stderr)
        return
    for c in collections:
        flag = "" if c.embedder == active else "  (different embedder)"
        print(f"{c.name}: {c.documents} documents, {c.chunks} chunks — {c.embedder}{flag}")


def cmd_docs(args: argparse.Namespace) -> None:
    with _kb(args) as kb:
        docs = kb.documents(args.collection)
    if args.json:
        print(json.dumps([asdict(d) for d in docs], indent=2))
        return
    for d in docs:
        pages = f", {d.total_pages} pages" if d.total_pages else ""
        print(f"[{d.collection}] {d.source_name} — {d.total_chunks} chunks{pages}\n    {d.source_path}")


def cmd_rm(args: argparse.Namespace) -> None:
    with _kb(args) as kb:
        removed = kb.remove(args.source, collection=args.collection)
    print(f"Removed {removed} document(s).", file=sys.stderr)
    if not removed:
        sys.exit(1)


def cmd_mcp(args: argparse.Namespace) -> None:
    from autorag.mcp_server import run

    run(db_path=args.db, embedder=args.embedder)


# --- parser -----------------------------------------------------------------------------


def _add_kb_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", help="Knowledge base file (default: $AUTORAG_HOME/knowledge.db, ~/.autorag)")
    p.add_argument(
        "--embedder",
        help="fastembed[:model] | ollama:<model>[@dims] | aegis (default: $AUTORAG_EMBEDDER or fastembed)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autorag",
        description="AutoRAG — local document ingestion and search for your apps and agents",
    )
    parser.add_argument("--version", action="version", version=f"AutoRAG {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("process", help="Convert documents into RAG-ready chunk folders")
    p.add_argument("files", nargs="+", help="Files or directories to process")
    p.add_argument("-o", "--output", default="./output", help="Output directory (default: ./output)")
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--strategy", choices=["hybrid", "hierarchical"], default="hybrid", help="Chunking strategy")
    p.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk (default: 512)")
    p.add_argument("--json", action="store_true", help="Output structured JSON per file to stdout")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bars")
    p.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="Verbose logging")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("info", help="Show file info without processing")
    p.add_argument("file", help="File to inspect")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("index", help="Add documents to the local knowledge base")
    p.add_argument("paths", nargs="+", help="Files or directories")
    p.add_argument("-c", "--collection", default="default", help="Collection name (default: default)")
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--tags", help="Comma-separated tags")
    p.add_argument("--force", action="store_true", help="Re-index files even if unchanged")
    p.add_argument("--json", action="store_true", help="One JSON line per file on stdout")
    p.add_argument("-q", "--quiet", action="store_true", help="No per-file progress")
    _add_kb_options(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="Search the knowledge base")
    p.add_argument("query")
    p.add_argument("-c", "--collection", help="Limit to one collection")
    p.add_argument("-k", type=int, default=5, help="Number of results (default: 5)")
    p.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    p.add_argument("--full", action="store_true", help="Print whole chunks")
    p.add_argument("--json", action="store_true", help="One JSON line per hit on stdout")
    _add_kb_options(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("collections", help="List collections")
    p.add_argument("--json", action="store_true")
    _add_kb_options(p)
    p.set_defaults(func=cmd_collections)

    p = sub.add_parser("docs", help="List indexed documents")
    p.add_argument("-c", "--collection", help="Limit to one collection")
    p.add_argument("--json", action="store_true")
    _add_kb_options(p)
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("rm", help="Remove a document from a collection")
    p.add_argument("source", help="Source path or file name")
    p.add_argument("-c", "--collection", default="default")
    _add_kb_options(p)
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("mcp", help="Run the MCP server on stdio (needs the [mcp] extra)")
    _add_kb_options(p)
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("gui", help="Launch the desktop GUI (needs the [gui] extra)")
    p.set_defaults(func=lambda _: _launch_gui())
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)
    _setup_logging(args.verbose)

    from autorag.embed import EmbeddingUnavailableError
    from autorag.store import EmbedderMismatchError

    try:
        args.func(args)
    except (EmbeddingUnavailableError, EmbedderMismatchError, KeyError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(2)


def _launch_gui() -> None:
    try:
        from autorag.gui import AutoRAGApp
    except ImportError as exc:
        print(f"The GUI needs the [gui] extra: pip install 'autorag[gui]' ({exc})", file=sys.stderr)
        sys.exit(1)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=str(log_dir / "autorag.log"),
        filemode="a",
    )
    app = AutoRAGApp()
    app.mainloop()


if __name__ == "__main__":
    main()
