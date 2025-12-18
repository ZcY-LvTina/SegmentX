"""Small download helper with progress + cancellation support."""

from __future__ import annotations

import os
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional


class DownloadCancelled(Exception):
    pass


ProgressCallback = Callable[[int, Optional[int]], None]


def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1024 * 256,
    timeout: float = 10.0,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    """Download a URL to dest with optional progress + cancel support.

    Raises DownloadCancelled when cancelled.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # type: ignore[arg-type]
            # Try to make read() interruptible by cancellation (short socket timeout).
            try:
                if hasattr(response, "fp") and hasattr(response.fp, "raw"):
                    response.fp.raw._sock.settimeout(1.0)  # type: ignore[attr-defined]
            except Exception:
                pass
            total = response.length or response.headers.get("Content-Length")
            total_size = int(total) if total else None

            downloaded = 0
            with open(dest, "wb") as fh:
                while True:
                    if cancel_event and cancel_event.is_set():
                        raise DownloadCancelled("cancelled by user")
                    try:
                        chunk = response.read(chunk_size)
                    except socket.timeout:
                        # allow cancellation check and continue
                        continue
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total_size)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network required
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network required
        raise RuntimeError(f"network error while downloading {url}: {exc.reason}") from exc

    if progress_cb:
        progress_cb(downloaded, total_size)
    return dest
