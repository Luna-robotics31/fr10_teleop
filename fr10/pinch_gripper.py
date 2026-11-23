"""Close and open the real FR10 gripper with a thumb-index pinch. No arm motion.

Three small parts, used by ``tools/pinch_gripper.py``:

* :class:`PinchDetector`   thumb-tip to index-tip distance -> pinched / not pinched
                           (hysteresis, a short hold time, tolerant of a lost frame);
* :class:`ScriptGripper`   the gripper calls of the standalone ``gripper.py``:
                           ``ActGripper(2, 1)`` then ``MoveGripper(2, 0 or 100, 100, 67, 2000, 0, ...)``;
* :class:`PinchGripperController`  decides when to send open / close and retries refused commands.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

from .config import Config
from .robots.base import RobotError
from .robots.fairino import describe_error, load_fairino_sdk

logger = logging.getLogger(__name__)

THUMB_TIP_JOINT = 4
INDEX_TIP_JOINT = 9


def pinch_distance(positions) -> Optional[float]:
    """Distance [m] between thumb tip and index fingertip, or ``None`` if the hand is not tracked."""
    points = np.asarray(positions, dtype=float)
    if points.shape != (25, 3) or not np.all(np.isfinite(points)):
        return None
    thumb, index = points[THUMB_TIP_JOINT], points[INDEX_TIP_JOINT]
    # An untracked hand reports every joint at the origin.
    if float(np.linalg.norm(thumb)) < 1e-6 and float(np.linalg.norm(index)) < 1e-6:
        return None
    return float(np.linalg.norm(thumb - index))


class PinchDetector:
    """Pinched while thumb and index tip stay close, with hysteresis and a short hold time.

    * A pinch starts when the distance is at or below ``on_m`` for ``hold_s``.
    * It ends when the distance is above ``off_m`` (or the hand is lost) for ``release_s``.
      Distances between ``on_m`` and ``off_m`` change nothing, so fingertip jitter is harmless.
    """

    def __init__(self, on_m: float = 0.03, off_m: float = 0.05, hold_s: float = 0.08, release_s: float = 0.10):
        if not 0.0 < on_m < off_m:
            raise ValueError("need 0 < on_m < off_m")
        if hold_s < 0.0 or release_s < 0.0:
            raise ValueError("hold_s and release_s must be >= 0")
        self.on_m, self.off_m, self.hold_s, self.release_s = on_m, off_m, hold_s, release_s
        self.pinched = False
        self._since: Optional[float] = None

    def update(self, distance: Optional[float], now: float) -> bool:
        """Feed one measurement (``None`` = hand not tracked); returns whether a pinch is active."""
        if not self.pinched:
            if distance is not None and distance <= self.on_m:
                if self._since is None:
                    self._since = now
                if now - self._since >= self.hold_s:
                    self.pinched, self._since = True, None
            else:
                self._since = None
        else:
            if distance is None or distance > self.off_m:
                if self._since is None:
                    self._since = now
                if now - self._since >= self.release_s:
                    self.pinched, self._since = False, None
            else:
                self._since = None
        return self.pinched


class PlainRpc:
    """``ActGripper`` / ``MoveGripper`` over plain XML-RPC on a socket of its own, with a timeout.

    The arguments are converted exactly as the vendor SDK converts them (integers, and ``rotNum`` as a
    float: the controller rejects an integer there). It opens no vendor-SDK client, so it can run next
    to a program that already holds the SDK connection, and a missing answer ends in an error after
    ``timeout`` seconds instead of waiting for ever. ``ActGripper`` gets a longer limit because
    activating a gripper takes seconds.
    """

    PORT = 20003

    def __init__(self, ip: str, timeout: float = 5.0, activation_timeout: float = 20.0, port: int = 20003):
        import xmlrpc.client

        def proxy(seconds: float):
            class _Transport(xmlrpc.client.Transport):
                def make_connection(self, host):
                    connection = super().make_connection(host)
                    connection.timeout = seconds
                    return connection

            return xmlrpc.client.ServerProxy(f"http://{ip}:{port}", transport=_Transport())

        self._proxy = proxy(timeout)  # moves
        self._slow_proxy = proxy(activation_timeout)  # activating a gripper takes seconds

    def _call(self, method: str, *args) -> int:
        try:
            result = getattr(self._slow_proxy if method == "ActGripper" else self._proxy, method)(*args)
        except Exception as exc:  # noqa: BLE001
            raise RobotError(f"{method}{args} got no answer from the controller: {exc}") from exc
        return int(result[0] if isinstance(result, (list, tuple)) else result)

    def ActGripper(self, index, action):
        return self._call("ActGripper", int(index), int(action))

    def MoveGripper(self, index, pos, vel, force, maxtime, block, type_, rot_num, rot_vel, rot_torque):
        return self._call(
            "MoveGripper", int(index), int(pos), int(vel), int(force), int(maxtime), int(block), int(type_),
            float(rot_num), int(rot_vel), int(rot_torque),
        )

    def CloseRPC(self):
        pass


class ScriptGripper:
    """The open / close functions of ``gripper.py``, through the vendor SDK (or plain XML-RPC).

    ``open_gripper()``  ->  ActGripper(index, 1); MoveGripper(index, 0,   100, 67, 2000, 0, 0, 0, 0, 0)
    ``close_gripper()`` ->  ActGripper(index, 1); MoveGripper(index, 100, 100, 67, 2000, 0, 0, 0, 0, 0)

    The numbers come from ``fairino.gripper`` in the configuration. ``block=0`` waits for the move
    to finish (up to ``max_time_ms``), as in ``gripper.py``; nothing else is running on this
    connection, so waiting is harmless here.
    """

    def __init__(self, cfg: Config, rpc=None):
        self.cfg = cfg
        self._rpc = rpc
        self.last_duration = 0.0
        # gripper.py activates before every move and waits for it. That is slow (an activation takes
        # seconds), so a program that follows a hand activates once (activate()) and then sends moves
        # that return at once: it sets activate_each_move = False and block = 1.
        self.activate_each_move = True
        self.block = 0

    @classmethod
    def connect(cls, cfg: Config, sdk=None) -> "ScriptGripper":
        """Connect to the controller exactly like ``connect_fairino.py``: ``Robot.RPC(ip)``."""
        sdk = sdk if sdk is not None else load_fairino_sdk(cfg.fairino.sdk_dir)
        if hasattr(sdk.RPC, "is_connect"):
            sdk.RPC.is_connect = True  # the SDK keeps "not connected" in a class attribute
        try:
            rpc = sdk.RPC(cfg.fairino.ip)
        except Exception as exc:  # noqa: BLE001 - vendor code
            raise RobotError(f"cannot connect to the controller at {cfg.fairino.ip}: {exc}") from exc
        if getattr(sdk.RPC, "is_connect", True) is False:
            raise RobotError(
                f"no answer from the controller at {cfg.fairino.ip}. Check the Ethernet cable and that this "
                "PC has an address in the same subnet (for example 192.168.58.10)."
            )
        return cls(cfg, rpc)

    @classmethod
    def connect_plain(cls, cfg: Config) -> "ScriptGripper":
        """Same calls over plain XML-RPC: no second vendor-SDK client next to a running teleoperation."""
        return cls(
            cfg,
            PlainRpc(
                cfg.fairino.ip, cfg.fairino.rpc_timeout_s, cfg.fairino.gripper.activation_timeout_s, cfg.fairino.rpc_port
            ),
        )

    def activate(self) -> None:
        """``ActGripper(index, 1)`` on its own. Takes seconds on a gripper that is not active yet."""
        if self._rpc is None:
            raise RobotError("not connected")
        gc = self.cfg.fairino.gripper
        started = time.perf_counter()
        code = self._rpc.ActGripper(gc.index, 1)
        if code != 0:
            raise RobotError(f"ActGripper({gc.index}, 1) answered {describe_error(code)}")
        logger.info("gripper %d activated in %.1f s", gc.index, time.perf_counter() - started)

    def close_connection(self) -> None:
        rpc, self._rpc = self._rpc, None
        if rpc is not None:
            try:
                rpc.CloseRPC()
            except Exception as exc:  # noqa: BLE001
                logger.debug("CloseRPC: %s", exc)

    def move_percent(
        self, percent: float, activate: Optional[bool] = None, block: Optional[int] = None, log: bool = True
    ) -> None:
        """Move the gripper to ``percent`` (0 = open ... 100 = closed) like ``gripper.py`` does.

        ``activate``: call ``ActGripper(index, 1)`` first (default: ``activate_each_move``).
        ``block``: 0 waits for the move to finish, 1 returns at once (default: ``self.block``).
        """
        gc = self.cfg.fairino.gripper
        if self._rpc is None:
            raise RobotError("not connected")
        activate = self.activate_each_move if activate is None else activate
        block = self.block if block is None else block
        position = int(round(percent))
        started = time.perf_counter()
        if activate:
            code = self._rpc.ActGripper(gc.index, 1)
            if code != 0:
                raise RobotError(f"ActGripper({gc.index}, 1) answered {describe_error(code)}")
        code = self._rpc.MoveGripper(
            gc.index, position, gc.speed_percent, gc.force_percent, gc.max_time_ms, block, 0, 0, 0, 0
        )
        if code != 0:
            raise RobotError(f"MoveGripper({gc.index}, {position}, ...) answered {describe_error(code)}")
        self.last_duration = time.perf_counter() - started
        if log:
            logger.info(
                "gripper %s: the controller accepted %sMoveGripper(%d %%) in %.0f ms",
                "CLOSE" if percent >= 50 else "OPEN", "ActGripper and " if activate else "", position,
                1e3 * self.last_duration,
            )

    def open_gripper(self) -> None:
        self.move_percent(self.cfg.fairino.gripper.open_percent)

    def close_gripper(self) -> None:
        self.move_percent(self.cfg.fairino.gripper.closed_percent)


class PinchGripperController:
    """Pinch closes the gripper, releasing the pinch opens it (or, with ``toggle``, each pinch flips it)."""

    def __init__(
        self,
        gripper,
        detector: Optional[PinchDetector] = None,
        toggle: bool = False,
        retry_s: float = 0.5,
        open_first: bool = False,
    ):
        self.gripper = gripper
        self.detector = detector or PinchDetector()
        self.toggle = toggle
        self.retry_s = retry_s
        self.state: Optional[str] = None  # "open" / "closed": what the gripper was last commanded to
        self.wanted: Optional[str] = "open" if open_first else None
        self.failures = 0
        self._last_attempt = -1e9
        self._was_pinched = False
        self.log: list = []  # (time, action, ok) of every command that was tried

    def update(self, distance: Optional[float], now: float) -> Optional[str]:
        """Process one measurement. Returns "CLOSE" / "OPEN" if a command was accepted this call."""
        pinched = self.detector.update(distance, now)
        if self.toggle:
            if pinched and not self._was_pinched:
                self.wanted = "open" if (self.wanted or self.state or "open") == "closed" else "closed"
        else:
            if pinched:
                self.wanted = "closed"
            elif self.state is not None or self._was_pinched:
                self.wanted = "open"  # a released pinch opens the gripper again
        self._was_pinched = pinched

        if self.wanted is None or self.wanted == self.state:
            return None
        if now - self._last_attempt < self.retry_s:
            return None
        self._last_attempt = now
        action = "CLOSE" if self.wanted == "closed" else "OPEN"
        try:
            (self.gripper.close_gripper if self.wanted == "closed" else self.gripper.open_gripper)()
        except RobotError as exc:
            self.failures += 1
            self.log.append((now, action, False))
            logger.warning("gripper %s failed (%d times): %s", action, self.failures, exc)
            return None
        self.failures = 0
        self.state = self.wanted
        self.log.append((now, action, True))
        logger.debug("gripper %s command done", action)
        return action


class ProportionalPinchController:
    """The gripper follows how far you pinch: a half pinch stops it halfway, a full pinch closes it.

    ``open_m`` (or more) thumb-index distance is open, ``closed_m`` (or less) is fully closed, and the
    positions in between map linearly onto 0..100 %. The distance is low-passed (0.1 s) and a new
    position is sent only when it changed by ``resolution`` percent, at most every ``min_interval_s``.
    A hand that is not tracked leaves the gripper where it is. Activation and waiting are the
    gripper's own settings (``ScriptGripper.activate_each_move`` / ``block``).

    Every gripper command goes to the same controller as the arm's joint commands, and a slow one can
    make the arm stutter. So few commands are sent: while the fingers are moving, one every
    ``min_interval_s`` and only for a change of ``resolution`` percent; once the fingers have been still
    for ``settle_s``, one final command to the exact position (if it is off by ``settle_resolution``).
    """

    def __init__(
        self,
        gripper,
        open_m: float = 0.08,
        closed_m: float = 0.02,
        resolution: float = 10.0,
        min_interval_s: float = 0.4,
        retry_s: float = 0.8,
        settle_s: float = 0.3,
        settle_resolution: float = 3.0,
        settled_only: bool = False,
        still_band: float = 1.0,
    ):
        """``settled_only``: send nothing while the fingers move, except fully open / closed (each move is
        expensive, e.g. it pauses the arm). ``still_band``: change in percent per update that still counts
        as "not moving" (hand tracking jitters by a few percent). ``min_interval_s`` is counted from the
        end of the previous command."""
        if not 0.0 < closed_m < open_m:
            raise ValueError("need 0 < closed_m < open_m")
        self.gripper = gripper
        self.open_m, self.closed_m = open_m, closed_m
        self.resolution, self.min_interval_s, self.retry_s = resolution, min_interval_s, retry_s
        self.settle_s, self.settle_resolution = settle_s, settle_resolution
        self.settled_only, self.still_band = settled_only, still_band
        self._previous_wanted: Optional[int] = None
        self._still_since: Optional[float] = None
        self.closure: Optional[float] = None  # low-passed 0..1
        self.position: Optional[int] = None  # the position last accepted by the controller
        self.failures = 0
        self._stamp: Optional[float] = None
        self._last_attempt = -1e9
        self.log: list = []

    def update(self, distance: Optional[float], now: float) -> Optional[int]:
        """Process one thumb-index distance; returns the position (0..100) if a command was accepted."""
        if distance is not None and np.isfinite(distance):
            raw = float(np.clip((self.open_m - distance) / (self.open_m - self.closed_m), 0.0, 1.0))
            if self.closure is None or self._stamp is None:
                self.closure = raw
            else:
                dt = max(0.0, now - self._stamp)
                self.closure += (1.0 - float(np.exp(-dt / 0.1))) * (raw - self.closure)
            self._stamp = now
        return self._follow(now)

    def update_closure(self, closure: Optional[float], now: float) -> Optional[int]:
        """Process a closure (0 open .. 1 closed) that the caller has already smoothed."""
        if closure is not None and np.isfinite(closure):
            self.closure = float(np.clip(closure, 0.0, 1.0))
        return self._follow(now)

    def resend(self) -> None:
        """Forget what was sent: the next update sends the present position again."""
        self.position = None
        self._last_attempt = -1e9

    def _follow(self, now: float) -> Optional[int]:
        if self.closure is None:
            return None
        wanted = int(round(100.0 * self.closure))
        at_end = wanted in (0, 100)
        if self._previous_wanted is None or abs(wanted - self._previous_wanted) > self.still_band:
            self._still_since = None  # the fingers are moving
            self._previous_wanted = wanted
        elif self._still_since is None:
            self._still_since = now
        settled = self._still_since is not None and now - self._still_since >= self.settle_s
        if self.settled_only and not (settled or at_end):
            return None
        needed = self.settle_resolution if settled else self.resolution
        if self.settled_only:
            needed = self.resolution  # every move is expensive: only a real change is worth one
        if self.position is not None and (
            wanted == self.position or (abs(wanted - self.position) < needed and not at_end)
        ):
            return None
        wait = self.retry_s if self.failures else self.min_interval_s
        if now - self._last_attempt < wait:
            return None
        self._last_attempt = now
        started = time.monotonic()
        try:
            self.gripper.move_percent(wanted, log=False)  # activation and waiting are the gripper's settings
        except RobotError as exc:
            self._last_attempt = now + (time.monotonic() - started)  # the interval counts from the end
            self.failures += 1
            self.log.append((now, wanted, False))
            logger.warning("gripper move to %d %% failed (%d times): %s", wanted, self.failures, exc)
            return None
        self._last_attempt = now + (time.monotonic() - started)  # the interval counts from the end
        self.failures = 0
        self.position = wanted
        self.log.append((now, wanted, True))
        return wanted
