"""
Tests for shared utilities.

The console-encoding helper matters more than it looks: without it, printing an
Arabic question on a Windows terminal raises ``UnicodeEncodeError`` and aborts
the process -- so the bilingual demo fails on precisely the feature it exists to
demonstrate.
"""
from __future__ import annotations

import io
import subprocess
import sys

from src.config.settings import PROJECT_ROOT
from src.utils.helpers import clean_text, configure_console_encoding

ARABIC = "ما هو سعر القائمة للمنتج SKU0001؟"


def test_configure_console_encoding_is_idempotent():
    """Entry points may call it more than once; that must be harmless."""
    configure_console_encoding()
    configure_console_encoding()


def test_configure_console_encoding_tolerates_unreconfigurable_streams(monkeypatch):
    """Under pytest stdout is a captured buffer; it must not raise."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    configure_console_encoding()


def test_arabic_survives_a_subprocess_stdout():
    """
    End-to-end guard: a child process must be able to print Arabic.

    Run as a subprocess because the encoding of *this* process's stdout is
    already fixed by pytest's capture, so an in-process assertion would pass
    even with the helper removed.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, r'%s');"
            "from src.utils.helpers import configure_console_encoding;"
            "configure_console_encoding();"
            "print(%r)" % (str(PROJECT_ROOT), ARABIC),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "SKU0001" in result.stdout
    assert "سعر" in result.stdout


def test_clean_text_collapses_whitespace():
    assert clean_text("  a\t b \n c  ") == "a b c"
