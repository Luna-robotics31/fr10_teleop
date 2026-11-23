"""Share the headset's hand data between programs on one PC.

The Quest connects to one headset server, so two programs cannot both own the headset. The
teleoperation program owns it and republishes the hand joint positions here; a second program
(``tools/pinch_gripper.py --from-teleop``) subscribes and needs no headset connection of its own.

One JSON message per update on a ZeroMQ PUB socket; the subscriber keeps only the newest.
Depends only on NumPy and pyzmq.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import zmq

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
SIDES = ("left", "right")


class HandBusPublisher:
    """Publishes ``{"left": positions|null, "right": positions|null, ages}`` at most ``rate_hz`` times a second."""

    def __init__(self, port: int, rate_hz: float = 30.0, bind_host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.bind(f"tcp://{bind_host}:{port}")
        except zmq.ZMQError as exc:
            self._socket.close(0)
            self._context.term()
            raise RuntimeError(f"cannot bind the hand data port {port}: {exc}") from exc
        self._period = 1.0 / rate_hz
        self._next = 0.0
        self.sent = 0

    def update(self, source, now: Optional[float] = None) -> bool:
        """Publish the hand data of ``source`` (needs ``hand_snapshot(side)``). Returns True if sent."""
        now = time.monotonic() if now is None else now
        if now < self._next or not hasattr(source, "hand_snapshot"):
            return False
        self._next = now + self._period
        message = {"v": PROTOCOL_VERSION, "stamp": time.time()}
        for side in SIDES:
            positions, age = source.hand_snapshot(side)
            message[side] = None if positions is None else np.round(np.asarray(positions, dtype=float), 5).tolist()
            message[f"{side}_age"] = None if age is None else float(age)
        try:
            self._socket.send(json.dumps(message, separators=(",", ":")).encode("utf-8"), zmq.NOBLOCK)
        except zmq.Again:
            return False
        self.sent += 1
        return True

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class HandBusSubscriber:
    """Newest hand data from a :class:`HandBusPublisher`, with the same ``hand_snapshot`` interface as the headset."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5560):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(f"tcp://{host}:{port}")
        self._latest: Optional[dict] = None
        self._received = 0.0
        self.messages = 0
        self.bad_messages = 0

    def _poll(self) -> None:
        while True:
            try:
                payload = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                data = json.loads(payload)
                if data.get("v") != PROTOCOL_VERSION:
                    raise ValueError("protocol version")
            except (ValueError, UnicodeDecodeError, AttributeError):
                self.bad_messages += 1
                continue
            self._latest, self._received = data, time.monotonic()
            self.messages += 1

    @property
    def connected(self) -> bool:
        """True while messages keep arriving (the teleoperation program is running)."""
        self._poll()
        return self._latest is not None and time.monotonic() - self._received < 1.0

    def hand_snapshot(self, side: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """``(positions (25, 3) or None, seconds since the headset last delivered data or None)``."""
        self._poll()
        if self._latest is None or time.monotonic() - self._received > 1.0:
            return None, None
        raw = self._latest.get(side)
        age = self._latest.get(f"{side}_age")
        if raw is None or age is None:
            return None, None
        try:
            positions = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return None, None
        return (positions if positions.shape == (25, 3) else None), float(age) + (time.monotonic() - self._received)

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()
