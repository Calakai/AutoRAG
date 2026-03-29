"""Command-line interface for AutoRAG."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from autorag import __version__


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _progress(message: str, fraction: float) -> None:
    bar_len = 30
    filled = int(bar_len * fraction)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stderr.write(f"\r  [{bar}] {fraction:3.0%} {message}")
    sys.stderr.flush()
    if fraction >= 1.0:
        sys.stderr.write("\n")


def cmd_process(args: argparse.Namespace) -> None:
    from autorag.config import ChunkingConfig, GeneralConfig, ProcessingConfig
    from autorag.pipeline import process_document
    from autorag.writer import write_output

    files = []
    for p in args.files:
        path = Path(p)
        if path.is_dir():
            # Collect all supported files from directory
            for ext in ("*.pdf", "*.epub", "*.docx", "*.txt", "*.md", "*.mobi"):
                files.extend(path.glob(ext))
        elif path.is_file():
            files.append(path)
        else:
            print(f"Warning: skipping {p} (not found)", file=sys.stderr)

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
        prefix = f"[{i + 1}/{len(files)}] {file_path.name}"
        print(prefix, file=sys.stderr)

        try:
            callback = _progress if not args.quiet else None
            result = process_document(file_path, config, progress_callback=callback)
            doc_dir = write_output(result, output_dir)
            total_chunks += result.total_chunks
            print(f"  -> {result.total_chunks} chunks in {result.processing_time_seconds}s", file=sys.stderr)

            if args.json:
                # Output structured result to stdout for piping
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
    import pymupdf

    from autorag.pipeline import _PYMUPDF_EXTENSIONS

    path = Path(args.file)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    ext = path.suffix.lower()
    info = {"file": str(path), "format": ext, "size_bytes": path.stat().st_size}

    if ext in _PYMUPDF_EXTENSIONS:
        doc = pymupdf.open(str(path))
        info["pages"] = len(doc)
        doc.close()
    elif ext == ".docx":
        info["format"] = "docx"
    elif ext in {".txt", ".md"}:
        info["format"] = "text"

    print(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autorag",
        description="AutoRAG — Convert documents into RAG-ready chunked output",
    )
    parser.add_argument("--version", action="version", version=f"AutoRAG {__version__}")
    sub = parser.add_subparsers(dest="command")

    # --- process ---
    p_proc = sub.add_parser("process", help="Process documents into RAG chunks")
    p_proc.add_argument("files", nargs="+", help="Files or directories to process")
    p_proc.add_argument("-o", "--output", default="./output", help="Output directory (default: ./output)")
    p_proc.add_argument("--strategy", choices=["hybrid", "hierarchical"], default="hybrid", help="Chunking strategy")
    p_proc.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk (default: 512)")
    p_proc.add_argument("--json", action="store_true", help="Output structured JSON per file to stdout (for piping)")
    p_proc.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bars")
    p_proc.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_proc.set_defaults(func=cmd_process)

    # --- info ---
    p_info = sub.add_parser("info", help="Show file info without processing")
    p_info.add_argument("file", help="File to inspect")
    p_info.set_defaults(func=cmd_info)

    # --- gui ---
    p_gui = sub.add_parser("gui", help="Launch the desktop GUI")
    p_gui.set_defaults(func=lambda _: _launch_gui())

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    _setup_logging(getattr(args, "verbose", False))
    args.func(args)


def _launch_gui() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=str(log_dir / "autorag.log"),
        filemode="a",
    )
    from autorag.gui import AutoRAGApp
    app = AutoRAGApp()
    app.mainloop()


if __name__ == "__main__":
    main()
