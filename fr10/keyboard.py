"""Non-blocking single-key input from the terminal (works locally and over SSH)."""

from __future__ import annotations

import logging
import os
import select
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class KeyListener:
    """Calls ``on_key(char)`` for every key pressed in the terminal.

    The terminal is switched to cbreak mode (no Enter needed, no echo) and restored on
    :meth:`stop`. If standard input is not a terminal the listener stays inactive.
    """

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fd: Optional[int] = None
        self._saved = None

    @property
    def active(self) -> bool:
        return self._thread is not None

    def start(self) -> bool:
        if self._thread is not None:
            return True
        try:
            import termios
            import tty

            if not sys.stdin.isatty():
                raise OSError("standard input is not a terminal")
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (ImportError, OSError, ValueError) as exc:
            logger.warning("keyboard control disabled (%s)", exc)
            self._fd = None
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="keyboard", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(self._fd, 32)
            except (OSError, ValueError):
                return
            if not data:  # end of file: terminal closed
                return
            for char in data.decode("utf-8", errors="ignore"):
                try:
                    self._on_key(char)
                except Exception:  # noqa: BLE001 - a bad handler must not kill the listener
                    logger.exception("key handler failed")

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        if self._fd is not None and self._saved is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001 - nothing more we can do while exiting
                pass
            self._saved = None
