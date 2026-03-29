"""CustomTkinter GUI for AutoRAG."""

from __future__ import annotations

import logging
import os
import platform
import queue
import subprocess
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from autorag.config import ChunkingConfig, GeneralConfig, MetadataConfig, ParsingConfig, ProcessingConfig
from autorag.pipeline import ProcessingCancelledError, process_document
from autorag.writer import write_output

logger = logging.getLogger(__name__)


class AutoRAGApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.title("AutoRAG")
        self.minsize(700, 500)
        self.geometry("750x540")

        self._selected_files: list[Path] = []
        self._output_path: Path | None = None
        self._progress_queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._start_time: float = 0.0
        self._indeterminate = False
        self._last_status_msg = "Ready"

        self._build_ui()

    def _build_ui(self) -> None:
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=15)

        # --- 1. File Input ---
        file_frame = ctk.CTkFrame(container)
        file_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(file_frame, text="Input Documents", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        file_row = ctk.CTkFrame(file_frame, fg_color="transparent")
        file_row.pack(fill="x", padx=10, pady=(0, 10))

        self._file_label = ctk.CTkLabel(file_row, text="No files selected", anchor="w")
        self._file_label.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(file_row, text="Browse", width=100, command=self._browse_files).pack(side="right")

        # --- 2. Output Directory ---
        out_frame = ctk.CTkFrame(container)
        out_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(out_frame, text="Output Directory", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        out_row = ctk.CTkFrame(out_frame, fg_color="transparent")
        out_row.pack(fill="x", padx=10, pady=(0, 10))

        self._output_var = ctk.StringVar(value="./output")
        ctk.CTkEntry(out_row, textvariable=self._output_var).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(out_row, text="Browse", width=100, command=self._browse_output).pack(side="right")

        # --- 3. Settings Panel ---
        settings_frame = ctk.CTkFrame(container)
        settings_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(settings_frame, text="Settings", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        settings_grid = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_grid.pack(fill="x", padx=10, pady=(0, 10))

        # Strategy dropdown
        ctk.CTkLabel(settings_grid, text="Chunking Strategy:").grid(row=0, column=0, sticky="w", pady=5)
        self._strategy_var = ctk.StringVar(value="Hybrid")
        ctk.CTkOptionMenu(
            settings_grid,
            variable=self._strategy_var,
            values=["Hybrid", "Hierarchical"],
            width=200,
            command=self._on_strategy_change,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=5)
        self._strategy_hint = ctk.CTkLabel(
            settings_grid,
            text="Splits by structure (headings, sections) with token limits. Best for most documents.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self._strategy_hint.grid(row=1, column=0, columnspan=2, sticky="w", padx=(5, 0), pady=(0, 5))

        # Max tokens dropdown
        ctk.CTkLabel(settings_grid, text="Chunk Size:").grid(row=2, column=0, sticky="w", pady=5)
        self._max_tokens_var = ctk.StringVar(value="512 (recommended)")
        ctk.CTkOptionMenu(
            settings_grid,
            variable=self._max_tokens_var,
            values=["128", "256", "512 (recommended)", "1024", "2048"],
            width=200,
        ).grid(row=2, column=1, sticky="w", padx=(10, 0), pady=5)
        ctk.CTkLabel(
            settings_grid,
            text="Smaller = more precise retrieval. Larger = more context per chunk.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=(5, 0), pady=(0, 5))

        # Info note
        ctk.CTkLabel(
            settings_grid,
            text="Supports PDF, EPUB, DOCX, TXT, MD, MOBI. Scanned pages auto-OCR'd.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # --- 4. Process Button ---
        self._process_btn = ctk.CTkButton(
            container,
            text="\u25b6  Process",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            command=self._on_process_click,
            state="disabled",
        )
        self._process_btn.pack(fill="x", pady=(0, 10))

        # --- 5. Progress Area ---
        progress_frame = ctk.CTkFrame(container)
        progress_frame.pack(fill="x", pady=(0, 10))

        self._status_label = ctk.CTkLabel(progress_frame, text="Ready", anchor="w")
        self._status_label.pack(fill="x", padx=10, pady=(10, 5))

        self._progress_bar = ctk.CTkProgressBar(progress_frame)
        self._progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self._progress_bar.set(0)

        # --- 6. Open Output Folder Button (hidden initially) ---
        self._open_btn = ctk.CTkButton(
            container,
            text="Open Output Folder",
            command=self._open_output_folder,
        )

    _STRATEGY_HINTS = {
        "Hybrid": "Splits by structure (headings, sections) with token limits. Best for most documents.",
        "Hierarchical": "Splits strictly by document headings. Best for well-structured documents with clear hierarchy.",
    }

    def _on_strategy_change(self, value: str) -> None:
        self._strategy_hint.configure(text=self._STRATEGY_HINTS.get(value, ""))

    # --- File Selection ---

    def _browse_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=[
                ("Supported files", "*.pdf *.epub *.docx *.txt *.md *.mobi *.xps *.fb2 *.cbz"),
                ("PDF", "*.pdf"),
                ("EPUB", "*.epub"),
                ("Word", "*.docx"),
                ("Text", "*.txt *.md"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self._selected_files = [Path(p) for p in paths]
            count = len(self._selected_files)
            if count == 1:
                name = self._selected_files[0].name
                info = self._file_info(self._selected_files[0])
                self._file_label.configure(text=f"{name}{info}")
            else:
                self._file_label.configure(text=f"{count} files selected")
            self._process_btn.configure(state="normal")

    @staticmethod
    def _file_info(path: Path) -> str:
        """Quick size/page info for display."""
        try:
            ext = path.suffix.lower()
            if ext in {".pdf", ".epub", ".xps", ".oxps", ".cbz", ".fb2", ".mobi"}:
                import pymupdf
                doc = pymupdf.open(str(path))
                info = f" ({len(doc)} pages)"
                doc.close()
                return info
            elif ext == ".docx":
                return " (Word document)"
            elif ext in {".txt", ".md", ".text", ".rst"}:
                return f" ({path.stat().st_size / 1024:.0f} KB)"
        except Exception:
            pass
        return ""

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self._output_var.set(path)

    # --- Processing ---

    def _build_config(self) -> ProcessingConfig:
        strategy_map = {"Hybrid": "hybrid", "Hierarchical": "hierarchical"}
        strategy = strategy_map.get(self._strategy_var.get(), "hybrid")

        try:
            max_tokens = int(self._max_tokens_var.get().split()[0])
        except (ValueError, IndexError):
            max_tokens = 512

        overlap = max(0, max_tokens // 8)

        return ProcessingConfig(
            general=GeneralConfig(output_dir=self._output_var.get()),
            parsing=ParsingConfig(),
            chunking=ChunkingConfig(
                strategy=strategy,
                max_tokens=max_tokens,
                overlap_tokens=overlap,
            ),
            metadata=MetadataConfig(),
        )

    def _on_process_click(self) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel_event.set()
            self._process_btn.configure(text="Cancelling...", state="disabled")
            return

        if not self._selected_files:
            return

        # Reset UI state
        self._cancel_event.clear()
        self._start_time = time.monotonic()
        self._last_status_msg = "Starting..."
        self._progress_bar.configure(mode="indeterminate")
        self._progress_bar.start()
        self._indeterminate = True
        self._status_label.configure(text="Starting...")
        self._open_btn.pack_forget()
        self._process_btn.configure(text="Cancel", state="normal")

        self._worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self._worker.start()
        self._poll_progress()
        self._tick_timer()

    def _run_pipeline(self) -> None:
        """Process all selected files sequentially on background thread."""
        try:
            config = self._build_config()
            output_dir = Path(config.general.output_dir)
            total_files = len(self._selected_files)
            total_chunks = 0
            failed: list[str] = []

            for file_idx, file_path in enumerate(self._selected_files):
                if self._cancel_event.is_set():
                    raise ProcessingCancelledError()

                prefix = f"[{file_idx + 1}/{total_files}] {file_path.name}: "

                def file_progress(msg: str, frac: float, _prefix=prefix, _idx=file_idx) -> None:
                    # Scale per-file progress into the overall progress range
                    overall = (_idx + frac) / total_files
                    self._thread_safe_progress(f"{_prefix}{msg}", overall)

                try:
                    result = process_document(
                        file_path=file_path,
                        config=config,
                        progress_callback=file_progress,
                        cancel_check=self._cancel_event.is_set,
                    )
                    write_output(result, output_dir)
                    total_chunks += result.total_chunks
                except ProcessingCancelledError:
                    raise
                except Exception as e:
                    logger.exception("Failed to process %s", file_path.name)
                    failed.append(f"{file_path.name}: {e}")

            # Build completion message
            if failed:
                fail_msg = f" ({len(failed)} failed)"
            else:
                fail_msg = ""
            self._progress_queue.put(("done", str(output_dir), total_chunks, total_files, fail_msg))

        except ProcessingCancelledError:
            self._progress_queue.put(("cancelled",))
        except Exception as e:
            logger.exception("Processing failed")
            self._progress_queue.put(("error", str(e)))

    def _thread_safe_progress(self, message: str, fraction: float) -> None:
        self._progress_queue.put(("progress", message, fraction))

    def _switch_to_determinate(self) -> None:
        if self._indeterminate:
            self._progress_bar.stop()
            self._progress_bar.configure(mode="determinate")
            self._indeterminate = False

    def _elapsed_str(self) -> str:
        elapsed = int(time.monotonic() - self._start_time)
        return f" ({elapsed}s)" if elapsed > 0 else ""

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self._progress_queue.get_nowait()
                if msg[0] == "progress":
                    fraction = msg[2]
                    if fraction >= 0.01:
                        self._switch_to_determinate()
                        self._progress_bar.set(fraction)
                    self._last_status_msg = msg[1]
                    self._status_label.configure(text=f"{msg[1]}{self._elapsed_str()}")
                elif msg[0] == "done":
                    self._switch_to_determinate()
                    self._output_path = Path(msg[1])
                    total_chunks = msg[2]
                    total_files = msg[3]
                    fail_msg = msg[4]
                    elapsed = int(time.monotonic() - self._start_time)
                    file_word = "file" if total_files == 1 else "files"
                    self._status_label.configure(
                        text=f"Done! {total_chunks} chunks from {total_files} {file_word} in {elapsed}s{fail_msg}"
                    )
                    self._progress_bar.set(1.0)
                    self._process_btn.configure(text="\u25b6  Process", state="normal")
                    self._open_btn.pack(fill="x", pady=(0, 10))
                elif msg[0] == "error":
                    self._switch_to_determinate()
                    self._status_label.configure(text=f"Error: {msg[1]}")
                    self._progress_bar.set(0)
                    self._process_btn.configure(text="\u25b6  Process", state="normal")
                    messagebox.showerror("Processing Error", msg[1])
                elif msg[0] == "cancelled":
                    self._switch_to_determinate()
                    self._status_label.configure(text="Processing cancelled.")
                    self._progress_bar.set(0)
                    self._process_btn.configure(text="\u25b6  Process", state="normal")
        except queue.Empty:
            pass

    def _poll_progress(self) -> None:
        self._drain_queue()
        if self._worker and self._worker.is_alive():
            self.after(100, self._poll_progress)
        else:
            self.after(50, self._drain_queue)

    def _tick_timer(self) -> None:
        """Update elapsed time every second so the UI never looks frozen."""
        if self._worker and self._worker.is_alive():
            self._status_label.configure(text=f"{self._last_status_msg}{self._elapsed_str()}")
            self.after(1000, self._tick_timer)

    # --- Output ---

    def _open_output_folder(self) -> None:
        if not self._output_path:
            return
        path = str(self._output_path)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
