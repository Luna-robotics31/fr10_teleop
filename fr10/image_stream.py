"""Camera frames over ZeroMQ: one JPEG image per message on a PUB socket.

This is the same wire format the ``teleimager`` server of xr_teleoperate uses for its ZMQ
streams, so :class:`JpegSubscriber` can read either the simulator camera (published with
:class:`JpegPublisher`) or a real camera served by ``teleimager-server``.

Depends only on NumPy, OpenCV and pyzmq (available in both conda environments).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)


class JpegPublisher:
    """Publishes BGR images as JPEG. Frames are dropped, never queued, if nobody keeps up."""

    def __init__(self, port: int, quality: int = 80, bind_host: str = "*"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.bind(f"tcp://{bind_host}:{port}")
        except zmq.ZMQError as exc:
            self._socket.close(0)
            self._context.term()
            raise RuntimeError(f"cannot bind image port {port}: {exc}") from exc
        self._params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    def publish(self, bgr: np.ndarray) -> bool:
        ok, buffer = cv2.imencode(".jpg", bgr, self._params)
        if not ok:
            return False
        try:
            self._socket.send(buffer.tobytes(), zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class JpegSubscriber:
    """Background receiver that always holds the most recent decoded frame."""

    def __init__(self, host: str, port: int):
        self._address = f"tcp://{host}:{port}"
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jpeg-subscriber", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        context = zmq.Context()
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self._address)
        try:
            while not self._stop.is_set():
                if not sock.poll(100, zmq.POLLIN):
                    continue
                try:
                    payload = sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    continue
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with self._lock:
                    self._frame = frame
                    self._stamp = time.monotonic()
                    self._count += 1
        except zmq.ZMQError as exc:  # context terminated or socket error
            logger.warning("image stream stopped: %s", exc)
        finally:
            sock.close(0)
            context.term()

    @property
    def frames_received(self) -> int:
        with self._lock:
            return self._count

    def latest(self, max_age: Optional[float] = None) -> Optional[np.ndarray]:
        """Most recent BGR frame, or ``None`` if there is none (or it is older than ``max_age``)."""
        with self._lock:
            if self._frame is None:
                return None
            if max_age is not None and time.monotonic() - self._stamp > max_age:
                return None
            return self._frame

    def latest_with_count(self) -> Tuple[Optional[np.ndarray], int]:
        """``(frame, running frame number)``; the number tells whether the frame is new."""
        with self._lock:
            return self._frame, self._count

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
