"""Backend-independent robot interface.

Every backend (mock, simulator, real FR10) is driven the same way:

    robot.connect()
    robot.start_streaming()            # set-points start at the measured position
    robot.set_joint_target(q)          # any rate, any thread
    robot.set_gripper(g)
    robot.stop_streaming()
    robot.close()

A dedicated thread turns the latest target into smooth set-points at the servo rate using
:class:`fr10.trajectory.JointSetpointFilter`, so simulation and hardware see exactly the
same motion profile.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..config import Config
from ..trajectory import JointSetpointFilter

logger = logging.getLogger(__name__)


class RobotError(RuntimeError):
    """The robot cannot do what was asked (connection, fault, invalid state)."""


@dataclass
class RobotState:
    stamp: float  # time.monotonic() at which this feedback was received
    q: np.ndarray  # measured joint positions [rad]
    dq: np.ndarray  # measured joint velocities [rad/s]
    gripper: Optional[float] = None  # measured closing fraction (0 open .. 1 closed) if known
    fault: Optional[str] = None  # human readable reason why the robot must not be commanded
    info: dict = field(default_factory=dict)  # backend specific extras

    @property
    def age(self) -> float:
        return time.monotonic() - self.stamp


class Robot(ABC):
    """Base class: owns the set-point filter and the streaming thread."""

    name = "robot"

    def __init__(self, cfg: Config, lower: Sequence[float], upper: Sequence[float]):
        self.cfg = cfg
        self.servo_period = 1.0 / cfg.control.servo_rate_hz
        self._filter = JointSetpointFilter(
            cfg.robot.max_velocity_rad_s,
            cfg.robot.max_acceleration_rad_s2,
            lower,
            upper,
            gain=cfg.control.tracking_gain,
        )
        self._lock = threading.Lock()  # guards the fields below and the filter
        self._target: Optional[np.ndarray] = None
        self._speed_scale = float(cfg.control.speed_scale)
        self._gripper_command = 0.0
        self._hold = False
        self._stream_fault: Optional[str] = None
        self._streaming = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self.max_lateness = 0.0  # worst servo tick lateness since streaming started [s]
        self.send_avg = 0.0  # running average duration of one set-point send [s]
        self.send_max = 0.0  # longest set-point send since streaming started [s]
        self._gate_since: Optional[float] = None
        self._lead_limit = math.radians(cfg.control.max_command_lead_deg)
        self._lead_soft = math.radians(cfg.control.soft_command_lead_deg)
        self._ticks = self._soft_ticks = self._hold_ticks = 0  # servo ticks, and those slowed / held back
        # Pausing the set-point stream on request (the FR10 controller takes a gripper move only while
        # the arm's stream is quiet): None, "braking" or "paused".
        self._pause_state: Optional[str] = None
        self._pause_since = 0.0
        self._pause_stop_point: Optional[np.ndarray] = None
        self._pause_ignored = False  # a request that outlived pause_timeout_s is ignored until withdrawn
        self._pause_withdrawn_at: Optional[float] = None  # request withdrawn, waiting for _pause_can_end()
        self.pause_timeout_s = 8.0  # longest pause for one request; then the stream goes on regardless
        self.pauses = 0  # number of pauses since streaming started
        # Latency guard: a backend that reports how many set-points wait in the robot's own queue
        # (_command_backlog) gets no new ones while more than this are waiting (None: no guard).
        self.max_backlog_points: Optional[int] = None
        self.drain_ticks = 0  # servo ticks skipped by the latency guard since streaming started
        self._was_draining = False
        self._draining = False  # above max_backlog_points: silent until the queue is (nearly) empty again
        self._backlog_window: list = []  # recent backlog readings, to trim a small persistent backlog

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if self._connected:
            return
        self._connect()
        self._connected = True
        logger.info("[%s] connected", self.name)

    def close(self) -> None:
        try:
            self.stop_streaming()
        finally:
            if self._connected:
                self._connected = False
                self._disconnect()
                logger.info("[%s] disconnected", self.name)

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ state
    def get_state(self) -> RobotState:
        """Latest feedback. Raises :class:`RobotError` if none is available."""
        if not self._connected:
            raise RobotError(f"{self.name}: not connected")
        state = self._read_state()
        if self._stream_fault and state.fault is None:
            state.fault = self._stream_fault
        return state

    def view_yaw(self) -> Optional[float]:
        """Heading of the operator's camera in the base frame, if the backend knows it."""
        return None

    # ------------------------------------------------------------------ streaming
    @property
    def streaming(self) -> bool:
        return self._streaming.is_set()

    def start_streaming(self) -> None:
        """Start sending set-points, beginning exactly at the measured joint position."""
        if self.streaming:
            return
        if not self._connected:
            raise RobotError(f"{self.name}: not connected")
        state = self._read_state()  # backend faults only; an old streaming fault ends here
        if state.fault:
            raise RobotError(f"{self.name}: cannot start, {state.fault}")
        if state.age > self.cfg.control.state_timeout_s:
            raise RobotError(f"{self.name}: cannot start, joint feedback is {state.age:.2f} s old")
        if not np.all(np.isfinite(state.q)):
            raise RobotError(f"{self.name}: cannot start, joint feedback is not finite")
        if state.gripper is not None and not np.isfinite(state.gripper):
            raise RobotError(f"{self.name}: cannot start, gripper feedback is not finite")
        if np.any(state.q < self._filter.lower - 1e-6) or np.any(state.q > self._filter.upper + 1e-6):
            raise RobotError(
                f"{self.name}: cannot start, the arm is outside the allowed joint range "
                f"(q = {np.round(np.rad2deg(state.q), 1).tolist()} deg). Jog it inside first."
            )
        with self._lock:
            self._filter.reset(state.q)
            self._target = state.q.copy()
            # Preserve the measured opening before the servo thread sends its first
            # command, including when reconnecting to a gripper holding an object.
            if state.gripper is not None:
                self._gripper_command = float(np.clip(state.gripper, 0.0, 1.0))
            self._stream_fault = None
            self._hold = False
        self.max_lateness = 0.0
        self.send_avg = self.send_max = 0.0
        self._gate_since = None
        self._ticks = self._soft_ticks = self._hold_ticks = 0
        self._pause_state, self._pause_stop_point, self._pause_ignored = None, None, False
        self._pause_withdrawn_at = None
        self.pauses = self.drain_ticks = 0
        self._draining, self._was_draining, self._backlog_window = False, False, []
        self._on_stream_start()
        self._streaming.set()
        self._thread = threading.Thread(target=self._stream_loop, name=f"{self.name}-servo", daemon=True)
        self._thread.start()
        logger.info("[%s] streaming set-points at %.0f Hz", self.name, self.cfg.control.servo_rate_hz)

    def stall_hint(self) -> str:
        """Extra advice appended when the arm does not follow its command (backend specific)."""
        return ""

    def gripper_status(self) -> str:
        """One line about the gripper thread for status messages (real robot only)."""
        return ""

    def resend_gripper(self) -> None:
        """Send the present gripper command again (only meaningful for the real gripper)."""

    def halt(self) -> None:
        """Stop commanding the arm immediately (backends may also stop motion on the controller)."""
        self.stop_streaming()

    def stop_streaming(self) -> None:
        was_streaming = self._streaming.is_set()
        self._streaming.clear()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("[%s] servo thread did not stop within 2 s", self.name)
        with self._lock:
            was_paused = self._pause_state == "paused"
            self._pause_state, self._pause_stop_point, self._pause_withdrawn_at = None, None, None
        if was_paused:
            self._set_paused(False)
        if was_streaming or thread is not None:
            try:
                self._on_stream_stop()
            except Exception as exc:  # noqa: BLE001 - shutting down, report and continue
                logger.warning("[%s] error while leaving streaming mode: %s", self.name, exc)

    def set_joint_target(self, q: Sequence[float]) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != self._filter.lower.shape or not np.all(np.isfinite(q)):
            raise ValueError("joint target must be a finite vector of 6 values")
        with self._lock:
            self._target = np.clip(q, self._filter.lower, self._filter.upper)

    def set_speed_scale(self, scale: float) -> None:
        with self._lock:
            self._speed_scale = float(np.clip(scale, 0.01, 1.0))

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    def set_gripper(self, closing: float) -> None:
        """0 = open, 1 = closed."""
        if not np.isfinite(closing):
            raise ValueError("gripper command must be finite")
        with self._lock:
            self._gripper_command = float(np.clip(closing, 0.0, 1.0))

    @property
    def gripper_command(self) -> float:
        return self._gripper_command

    def commanded_position(self) -> np.ndarray:
        """The set-point most recently sent (the measured position before streaming starts)."""
        with self._lock:
            if self._filter.initialized:
                return self._filter.position
        return self.get_state().q.copy()

    def joint_target(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._target is None else self._target.copy()

    def resync(self) -> np.ndarray:
        """Restart the set-point filter at the measured position (after a reset or a stall)."""
        state = self.get_state()
        if not np.all(np.isfinite(state.q)):
            raise RobotError(f"{self.name}: joint feedback is not finite")
        q = np.clip(state.q, self._filter.lower, self._filter.upper)
        with self._lock:
            self._filter.reset(q)
            self._target = q.copy()
        return q

    def clear_fault(self) -> None:
        """Try to clear a fault so that streaming can be started again."""
        self._stream_fault = None
        self._clear_fault()

    def reset_scene(self) -> bool:
        """Put the scene back to its initial state (simulation only). Returns False if unsupported."""
        return False

    def lead_stats(self) -> tuple:
        """(% of servo ticks slowed because the arm lags the command, % held back) since streaming started."""
        if self._ticks == 0:
            return 0.0, 0.0
        return 100.0 * self._soft_ticks / self._ticks, 100.0 * self._hold_ticks / self._ticks

    @property
    def lag_hold_time(self) -> float:
        """Seconds the command has been held back because the arm lags it by the maximum lead."""
        since = self._gate_since
        return 0.0 if since is None else time.monotonic() - since

    @property
    def paused(self) -> bool:
        """True while the stream is paused (or braking for a pause) on the backend's request."""
        return self._pause_state is not None

    # Braking for a pause ends when every joint is this slow and this close to its stopping point.
    _PAUSE_REST_VELOCITY = math.radians(0.3)  # [rad/s]
    _PAUSE_REST_DISTANCE = math.radians(0.05)  # [rad]
    _PAUSE_MAX_BRAKING_S = 1.5

    def _pause_step(self, wanted: bool, period: float):
        """Pause handling for one servo tick (called with ``self._lock`` held).

        Returns ``(handled, q_cmd, notify)``: ``handled`` means this tick belongs to a pause and
        ``q_cmd`` is what to send (``None``: send nothing); ``notify`` is True/False when the backend
        must now be told that the stream is paused / running again.
        """
        now = time.monotonic()
        if not wanted:
            self._pause_ignored = False
            if self._pause_state is None:
                return False, None, None
            if self._pause_state == "paused":
                # The request is withdrawn, but the backend may still need silence (on the FR10 the
                # gripper move it just sent runs in the controller's queue: set-points sent now would
                # pile up behind it and make the arm lag for the rest of the session).
                if self._pause_withdrawn_at is None:
                    self._pause_withdrawn_at = now
                if (
                    not self._pause_can_end(now - self._pause_withdrawn_at)
                    and now - self._pause_since <= self.pause_timeout_s
                ):
                    return True, None, None
            # Resume: from rest at the last set-point, which the arm has reached during the pause.
            was_paused = self._pause_state == "paused"
            self._pause_state, self._pause_stop_point, self._pause_withdrawn_at = None, None, None
            self._filter.stop()
            self._gate_since = None
            return False, None, (False if was_paused else None)
        if self._pause_ignored:
            return False, None, None
        if self._pause_withdrawn_at is not None:  # a new request while still silent: a fresh pause
            self._pause_withdrawn_at, self._pause_since = None, now
        if self._pause_state is None:
            # Brake smoothly: aim at the point where the present motion can come to rest.
            self._pause_state, self._pause_since = "braking", now
            self._pause_stop_point = self._filter.stopping_point(self._speed_scale)
            self.pauses += 1
        if now - self._pause_since > self.pause_timeout_s:
            logger.warning(
                "[%s] paused for %.1f s and the request is still there: moving on", self.name, now - self._pause_since
            )
            was_paused = self._pause_state == "paused"
            self._pause_state, self._pause_stop_point, self._pause_ignored = None, None, True
            self._pause_withdrawn_at = None
            self._filter.stop()
            return False, None, (False if was_paused else None)
        if self._pause_state == "braking":
            q_cmd = self._filter.step(self._pause_stop_point, period, self._speed_scale)
            at_rest = (
                float(np.max(np.abs(self._filter.velocity))) < self._PAUSE_REST_VELOCITY
                and float(np.max(np.abs(q_cmd - self._pause_stop_point))) < self._PAUSE_REST_DISTANCE
            )
            if at_rest or now - self._pause_since > self._PAUSE_MAX_BRAKING_S:
                self._filter.stop()
                self._pause_state = "paused"
                return True, q_cmd, True  # this last set-point is still sent; nothing after it
            return True, q_cmd, None
        return True, None, None  # paused: the stream is silent

    # Trim: a backlog of at least this many set-points in every one of the last _TRIM_WINDOW readings is
    # persistent (not jitter); one tick is then skipped, which removes one queued set-point.
    _TRIM_POINTS = 2
    _TRIM_WINDOW = 10

    def _latency_guard(self, backlog: int) -> bool:
        """True if this tick must send nothing because the robot's own queue holds set-points.

        Above ``max_backlog_points``: silent until the queue is down to 1 (hysteresis). Below it, a small
        backlog that persists for ``_TRIM_WINDOW`` ticks is trimmed one tick at a time. Set-points then
        arrive exactly as fast as they run, so without this a backlog (= arm lag) would never shrink.
        """
        if self.max_backlog_points is None or self._pause_state is not None:
            self._backlog_window.clear()
            return False
        if backlog > self.max_backlog_points:
            self._draining = True
        elif self._draining and backlog <= 1:
            self._draining = False
        if self._draining:
            self._backlog_window.clear()
            return True
        self._backlog_window.append(backlog)
        if len(self._backlog_window) > self._TRIM_WINDOW:
            del self._backlog_window[0]
        if len(self._backlog_window) == self._TRIM_WINDOW and min(self._backlog_window) >= self._TRIM_POINTS:
            self._backlog_window.clear()
            return True
        return False

    def _stream_loop(self) -> None:
        period = self.servo_period
        next_tick = time.perf_counter()
        try:
            while self._streaming.is_set():
                try:
                    measured = self._read_state().q
                except Exception:  # noqa: BLE001 - without feedback the supervisor will report it
                    measured = None
                pause_wanted = self._pause_requested()
                backlog = self._command_backlog() if self.max_backlog_points is not None else 0
                skip = self._latency_guard(backlog)
                draining = skip and self._draining
                with self._lock:
                    handled, pause_q, notify = self._pause_step(pause_wanted, period)
                    if handled:
                        q_cmd = pause_q
                    elif self._hold:
                        q_cmd = None
                    elif skip:
                        # Latency guard: the robot still has set-points queued; adding more would only make
                        # it lag. Send nothing this tick and keep the filter where it is (no jump later).
                        q_cmd = None
                        self.drain_ticks += 1
                        if draining and self._gate_since is None:
                            self._gate_since = time.monotonic()  # a queue that never drains is a stall
                    else:
                        lead = (
                            0.0 if measured is None
                            else float(np.max(np.abs(self._filter.position - measured)))
                        )
                        # Soft limit: no effect up to _lead_soft, then the velocity limit falls to zero at
                        # _lead_limit. Braking is untouched, so the motion stays smooth.
                        velocity_scale = (
                            1.0 if lead <= self._lead_soft
                            else (self._lead_limit - lead) / (self._lead_limit - self._lead_soft)
                        )
                        self._ticks += 1
                        if velocity_scale < 0.05:
                            # The arm is far behind (blocked?): keep sending the same set-point.
                            self._hold_ticks += 1
                            self._filter.stop()
                            q_cmd = self._filter.position
                            if self._gate_since is None:
                                self._gate_since = time.monotonic()
                        else:
                            if velocity_scale < 1.0:
                                self._soft_ticks += 1
                            self._gate_since = None
                            q_cmd = self._filter.step(self._target, period, self._speed_scale, velocity_scale)
                    gripper = self._gripper_command
                if draining and not self._was_draining:
                    logger.warning(
                        "[%s] the robot has %d set-points queued (%.0f ms of lag): waiting for them to run",
                        self.name, backlog, 1e3 * backlog * period,
                    )
                self._was_draining = draining
                silent = (handled or skip) and q_cmd is None
                if not silent:
                    started = time.perf_counter()
                    self._send_setpoint(q_cmd, gripper)
                    took = time.perf_counter() - started
                    self.send_avg += 0.02 * (took - self.send_avg)
                    self.send_max = max(self.send_max, took)
                if notify is not None:
                    self._set_paused(notify)
                    if notify:
                        logger.info("[%s] arm stopped and paused for the gripper", self.name)
                    else:
                        logger.info("[%s] arm follows again", self.name)
                        next_tick = time.perf_counter()  # no burst to catch up after the pause
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    self.max_lateness = max(self.max_lateness, -delay)
                    if delay < -5.0 * period:  # far behind: do not burst to catch up
                        next_tick = time.perf_counter()
        except Exception as exc:  # noqa: BLE001 - any failure must stop motion and be reported
            self._stream_fault = f"set-point streaming failed: {exc}"
            logger.error("[%s] %s", self.name, self._stream_fault)
            self._streaming.clear()

    # ------------------------------------------------------------------ backend hooks
    @abstractmethod
    def _connect(self) -> None: ...

    @abstractmethod
    def _disconnect(self) -> None: ...

    @abstractmethod
    def _read_state(self) -> RobotState: ...

    @abstractmethod
    def _send_setpoint(self, q: Optional[np.ndarray], gripper: float) -> None:
        """Send one servo set-point. ``q`` is ``None`` while the backend must hold position."""

    def _on_stream_start(self) -> None:
        """Called before the first set-point (enter servo mode)."""

    def _on_stream_stop(self) -> None:
        """Called after the last set-point (leave servo mode)."""

    def _pause_requested(self) -> bool:
        """True while the backend wants the set-point stream to brake and fall silent (checked every tick)."""
        return False

    def _set_paused(self, paused: bool) -> None:
        """Tells the backend that the stream is now silent (True) or running again (False)."""

    def _pause_can_end(self, since_withdrawn: float) -> bool:
        """After the request is withdrawn: True once the stream may run again (default: at once)."""
        return True

    def _command_backlog(self) -> int:
        """Set-points waiting in the robot's own queue (only used when max_backlog_points is set)."""
        return 0

    def _clear_fault(self) -> None:
        """Backend specific fault reset."""
