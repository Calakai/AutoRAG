"""Double-click this file to launch AutoRAG GUI (no console window)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from autorag.cli import _launch_gui

_launch_gui()
