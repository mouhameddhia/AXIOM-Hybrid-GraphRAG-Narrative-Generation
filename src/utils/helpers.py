"""
Shared utilities.

Keep this module for genuinely generic helpers. Anything specific to retrieval,
the graph, or the pipeline belongs in its own module.
"""
from __future__ import annotations

import logging
import sys
from typing import IO, Any

__all__ = ["clean_text", "configure_console_encoding", "setup_logging"]


def configure_console_encoding(errors: str = "replace") -> None:
    """
    Force UTF-8 on stdout/stderr so bilingual output survives the terminal.

    On Windows the console defaults to a legacy code page (cp1252 on this
    machine). Printing an Arabic question then raises ``UnicodeEncodeError``
    and kills the process -- which, for a system whose brief is bilingual
    English/Arabic explanations, means the demo dies on exactly the feature it
    is meant to show.

    ``errors="replace"`` is deliberate: on a terminal that genuinely cannot
    render Arabic glyphs, showing replacement characters is better than
    aborting the run. Redirected output (a file or a pipe) is unaffected and
    still receives correct UTF-8.

    Safe to call repeatedly, and a no-op on streams that cannot be
    reconfigured (a captured buffer under pytest, for instance).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors=errors)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            # Already detached, or a stream that does not support it.
            continue


def setup_logging(level: int = logging.INFO, *, stream: IO[Any] | None = None) -> None:
    """
    Configure logging for a command-line entry point.

    Also fixes console encoding, since every entry point calls this and log
    records can themselves carry non-ASCII text.
    """
    configure_console_encoding()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=stream,
        force=True,  # replace any handler a library installed on import
    )


def clean_text(text: str) -> str:
    """Collapse whitespace. Used before indexing or display."""
    return " ".join(text.split()).strip()
