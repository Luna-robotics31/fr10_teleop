"""Wire protocol between the teleoperation program and the simulator.

Two ZeroMQ PUB/SUB channels carry single-frame JSON messages; the simulator binds both:

    teleop  --ArmCommand-->  simulator      tcp://<sim host>:<command_port>
    teleop  <--ArmState----  simulator      tcp://<sim host>:<state_port>

Both receivers keep only the newest message (``ZMQ_CONFLATE``): this is a real-time stream,
stale set-points are worthless.

This module is imported by both conda environments (teleop: Python 3.10, simulator: Python
3.11) and therefore depends only on the standard library, NumPy and pyzmq.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import zmq

PROTOCOL_VERSION = 1
NUM_JOINTS = 6


class ProtocolError(ValueError):
    """A received message is malformed or from an incompatible version."""


def _floats(values, count: int, name: str) -> List[float]:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"'{name}' is not a list of numbers") from exc
    if len(out) != count or not all(math.isfinite(v) for v in out):
        raise ProtocolError(f"'{name}' must contain {count} finite numbers")
    return out


def _decode(payload: bytes, kind: str) -> dict:
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError("message is not valid JSON") from exc
    if not isinstance(data, dict) or data.get("type") != kind:
        raise ProtocolError(f"expected a '{kind}' message")
    if data.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {data.get('v')} != {PROTOCOL_VERSION}; update both programs")
    return data


@dataclass
class ArmCommand:
    """Set-point sent to the simulator at the servo rate."""

    seq: int
    stamp: float  # sender wall-clock time [s]
    q: Optional[List[float]]  # arm joint targets [rad]; None = keep the current target
    gripper: Optional[float] = None  # 0 = open ... 1 = closed; None = keep
    reset_id: int = 0  # the simulator resets the scene whenever this number changes

    def encode(self) -> bytes:
        return json.dumps(
            {
                "type": "cmd",
                "v": PROTOCOL_VERSION,
                "seq": int(self.seq),
                "stamp": float(self.stamp),
                "q": None if self.q is None else [float(v) for v in self.q],
                "gripper": None if self.gripper is None else float(self.gripper),
                "reset_id": int(self.reset_id),
            },
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def decode(payload: bytes) -> "ArmCommand":
        data = _decode(payload, "cmd")
        try:
            q = None if data["q"] is None else _floats(data["q"], NUM_JOINTS, "q")
            gripper = data.get("gripper")
            if gripper is not None:
                gripper = float(gripper)
                if not math.isfinite(gripper):
                    raise ProtocolError("'gripper' must be finite")
                gripper = min(1.0, max(0.0, gripper))
            return ArmCommand(int(data["seq"]), float(data["stamp"]), q, gripper, int(data.get("reset_id", 0)))
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ProtocolError):
                raise
            raise ProtocolError(f"malformed command: {exc}") from exc


@dataclass
class ArmState:
    """Feedback published by the simulator after every physics step."""

    seq: int
    stamp: float  # simulator wall-clock time [s]
    sim_time: float  # simulated time [s]
    q: List[float]  # measured arm joint positions [rad]
    dq: List[float]  # measured arm joint velocities [rad/s]
    gripper: float  # measured closing fraction, 0 = open ... 1 = closed
    cmd_seq: int = -1  # sequence number of the last command applied (-1: none yet)
    reset_id: int = 0  # last scene reset that was executed
    view_yaw: float = 0.0  # heading of the operator camera in the robot base frame [rad]
    objects: Dict[str, List[float]] = field(default_factory=dict)  # name -> [x, y, z, qw, qx, qy, qz]

    def encode(self) -> bytes:
        return json.dumps(
            {
                "type": "state",
                "v": PROTOCOL_VERSION,
                "seq": int(self.seq),
                "stamp": float(self.stamp),
                "sim_time": float(self.sim_time),
                "q": [float(v) for v in self.q],
                "dq": [float(v) for v in self.dq],
                "gripper": float(self.gripper),
                "cmd_seq": int(self.cmd_seq),
                "reset_id": int(self.reset_id),
                "view_yaw": float(self.view_yaw),
                "objects": {str(k): [float(x) for x in v] for k, v in self.objects.items()},
            },
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def decode(payload: bytes) -> "ArmState":
        data = _decode(payload, "state")
        try:
            return ArmState(
                seq=int(data["seq"]),
                stamp=float(data["stamp"]),
                sim_time=float(data["sim_time"]),
                q=_floats(data["q"], NUM_JOINTS, "q"),
                dq=_floats(data["dq"], NUM_JOINTS, "dq"),
                gripper=float(data["gripper"]),
                cmd_seq=int(data.get("cmd_seq", -1)),
                reset_id=int(data.get("reset_id", 0)),
                view_yaw=float(data.get("view_yaw", 0.0)),
                objects={str(k): [float(x) for x in v] for k, v in dict(data.get("objects", {})).items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ProtocolError):
                raise
            raise ProtocolError(f"malformed state: {exc}") from exc


def _latest_only_subscriber(context: zmq.Context) -> zmq.Socket:
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)  # must be set before bind/connect
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    return sock


def _publisher(context: zmq.Context) -> zmq.Socket:
    sock = context.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.setsockopt(zmq.LINGER, 0)
    return sock


class SimServerLink:
    """Simulator side: publishes :class:`ArmState`, receives :class:`ArmCommand`."""

    def __init__(self, state_port: int, command_port: int, bind_host: str = "*"):
        self._context = zmq.Context()
        self._state_pub = _publisher(self._context)
        self._command_sub = _latest_only_subscriber(self._context)
        try:
            self._state_pub.bind(f"tcp://{bind_host}:{state_port}")
            self._command_sub.bind(f"tcp://{bind_host}:{command_port}")
        except zmq.ZMQError as exc:
            self.close()
            raise RuntimeError(
                f"cannot bind ports {state_port}/{command_port} ({exc}); is another simulator already running?"
            ) from exc
        self.bad_messages = 0

    def publish_state(self, state: ArmState) -> None:
        try:
            self._state_pub.send(state.encode(), zmq.NOBLOCK)
        except zmq.Again:
            pass  # a slow subscriber must never stall the physics loop

    def poll_command(self) -> Optional[ArmCommand]:
        """Newest command received since the last call, or ``None``."""
        try:
            payload = self._command_sub.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        try:
            return ArmCommand.decode(payload)
        except ProtocolError:
            self.bad_messages += 1
            return None

    def close(self) -> None:
        for sock in (getattr(self, "_state_pub", None), getattr(self, "_command_sub", None)):
            if sock is not None:
                sock.close(0)
        self._context.term()


class SimClientLink:
    """Teleop side: sends :class:`ArmCommand`, receives :class:`ArmState`.

    ``send_command`` and ``receive_state`` may be called from different threads, but each of
    them from one thread at a time (ZeroMQ sockets are not thread-safe).
    """

    def __init__(self, host: str, state_port: int, command_port: int):
        self._context = zmq.Context()
        self._state_sub = _latest_only_subscriber(self._context)
        self._command_pub = _publisher(self._context)
        self._state_sub.connect(f"tcp://{host}:{state_port}")
        self._command_pub.connect(f"tcp://{host}:{command_port}")
        self.bad_messages = 0

    def send_command(self, command: ArmCommand) -> None:
        try:
            self._command_pub.send(command.encode(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def receive_state(self, timeout_ms: int = 100) -> Optional[ArmState]:
        """Wait up to ``timeout_ms`` for the newest state."""
        if not self._state_sub.poll(timeout_ms, zmq.POLLIN):
            return None
        try:
            return ArmState.decode(self._state_sub.recv(zmq.NOBLOCK))
        except zmq.Again:
            return None
        except ProtocolError:
            self.bad_messages += 1
            return None

    def close(self) -> None:
        self._state_sub.close(0)
        self._command_pub.close(0)
        self._context.term()
