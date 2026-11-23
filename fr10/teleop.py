"""Teleoperation state machine: operator input -> IK -> robot.

Modes
-----
IDLE      connected, no set-points are sent. ``start`` -> READY.
READY     set-points are streamed, the arm holds. Closing the clutch -> TRACKING.
TRACKING  the tool follows the operator's hand. Opening the clutch -> READY.
HOMING    the arm moves to the home pose; closing the clutch cancels.
FAULT     something is wrong (robot fault, lost feedback, arm not following). Streaming is
          stopped; ``start`` clears the fault and returns to READY.

The clutch is a dead-man switch: the arm only moves while the operator actively holds it,
and after any interruption it must be released and closed again.
"""

from __future__ import annotations

import enum
import logging
import math
import queue
import time
from typing import Optional

import numpy as np

from .config import Config
from .kinematics import IKSolver, RobotModel
from .mapping import WorkspaceMapper
from .math_utils import pose_to_xyzquat
from .robots.base import Robot, RobotError, RobotState
from .trajectory import JointMove
from .xr_input import InputSource, OperatorInput

logger = logging.getLogger(__name__)

EVENTS = ("start", "stop", "open_gripper", "close_gripper", "home", "record", "reset", "quit", "faster", "slower")


class Mode(enum.Enum):
    IDLE = "idle"
    READY = "ready"
    TRACKING = "tracking"
    HOMING = "homing"
    FAULT = "fault"


class Teleoperator:
    def __init__(
        self,
        cfg: Config,
        robot: Robot,
        model: RobotModel,
        ik: IKSolver,
        source: InputSource,
        recorder=None,
        images=None,
        viewer=None,
        hand_publisher=None,
    ):
        self.cfg = cfg
        self.robot = robot
        self.model = model
        self.ik = ik
        self.source = source
        self.recorder = recorder
        self.images = images
        self.viewer = viewer
        self.hand_publisher = hand_publisher  # shares the hand data with tools/pinch_gripper.py
        # "palm down" must give the home orientation of the gripper (pointing down).
        self.mapper = WorkspaceMapper(cfg.mapping, hand=cfg.xr.hand, neutral_rotation=model.fk(model.home)[:3, :3])
        self._last_align_log = 0.0
        self._hand_mode = cfg.xr.input_mode == "hand"
        self._gripper_needs_sync = False
        self._sync_range = None  # (min, max) of the hand closure seen since the sync started
        self._tcp_at_engage = None

        self.mode = Mode.IDLE
        self.fault_reason: Optional[str] = None
        self.stopped = False
        self._events: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._period = 1.0 / cfg.control.rate_hz
        self._last_step: Optional[float] = None
        self._q_ref: Optional[np.ndarray] = None  # joint target = seed of the next IK solve
        self._clutch_armed = False
        self._home_move: Optional[JointMove] = None
        self._prev_a = False
        self._prev_b = False
        self._gripper = 0.0
        self._gripper_desired = 0.0
        self._toggle_closed = False
        self._prev_grip_high = False
        self._lag_since: Optional[float] = None
        self._last_frame = -1
        self._last_status = 0.0
        self._last_viewer = 0.0
        self._connected_logged = False
        self._ik_fail_logged = 0.0
        # diagnostics of the latest cycle
        self.last_input = OperatorInput()
        self.last_state: Optional[RobotState] = None
        self.last_ik_error = (0.0, 0.0)
        self.workspace_limited = False

    # ------------------------------------------------------------------ public API
    def post(self, event: str) -> None:
        """Request an action (thread-safe): one of ``EVENTS``."""
        if event not in EVENTS:
            raise ValueError(f"unknown event '{event}'")
        self._events.put(event)

    @property
    def joint_reference(self) -> Optional[np.ndarray]:
        return None if self._q_ref is None else self._q_ref.copy()

    def run(self, duration: Optional[float] = None) -> None:
        """Run until ``quit`` is requested, or for a bounded number of seconds."""
        if duration is not None and (not math.isfinite(duration) or duration <= 0.0):
            raise ValueError("duration must be a finite positive number of seconds")
        if self.cfg.control.auto_start:
            self.post("start")
        next_tick = time.perf_counter()
        deadline = None if duration is None else next_tick + duration
        try:
            while not self.stopped and (deadline is None or time.perf_counter() < deadline):
                self.step()
                next_tick += self._period
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -5.0 * self._period:
                    next_tick = time.perf_counter()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Stop commanding the robot. The arm stays where it is."""
        self.mapper.release()
        try:
            self.robot.stop_streaming()
        except Exception as exc:  # noqa: BLE001
            logger.warning("while stopping the robot: %s", exc)
        if self.recorder is not None and self.recorder.active:
            self.recorder.stop()
        if self.mode is not Mode.FAULT:
            self.mode = Mode.IDLE

    def step(self) -> None:
        """One control cycle."""
        now = time.monotonic()
        dt = self._period if self._last_step is None else float(np.clip(now - self._last_step, 1e-3, 0.1))
        self._last_step = now

        inp = self.source.read()
        if self.hand_publisher is not None:
            self.hand_publisher.update(self.source, now)
        self.last_input = inp
        if inp.connected and not self._connected_logged:
            self._connected_logged = True
            logger.info("headset connected, motion data is arriving")
        self._handle_buttons(inp)
        self._handle_events()

        if self.mode in (Mode.READY, Mode.TRACKING, Mode.HOMING):
            state = self._supervise(now)
            if state is not None:
                if self.mode is Mode.READY:
                    self._step_ready(inp)
                elif self.mode is Mode.TRACKING:
                    self._step_tracking(inp, dt)
                else:
                    self._step_homing(inp, dt)
                self._update_gripper(inp, dt)
                self._record(now, state)
        else:
            try:
                self.last_state = self.robot.get_state()
            except RobotError:
                self.last_state = None

        self._update_display(now)
        self._log_status(now)

    # ------------------------------------------------------------------ events
    def _handle_buttons(self, inp: OperatorInput) -> None:
        if inp.button_a and not self._prev_a and self.recorder is not None:
            self._events.put("record")
        if inp.button_b and not self._prev_b:
            self._events.put("home")
        self._prev_a, self._prev_b = inp.button_a, inp.button_b

    def _handle_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if event == "quit":
                self.stopped = True
            elif event == "start":
                self._on_start()
            elif event == "stop":
                self._on_stop()
            elif event == "open_gripper":
                self._on_gripper_key(0.0)
            elif event == "close_gripper":
                self._on_gripper_key(1.0)
            elif event == "home":
                self._on_home()
            elif event == "reset":
                self._on_reset()
            elif event == "record":
                self._on_record()
            elif event in ("faster", "slower"):
                step = 0.1 if event == "faster" else -0.1
                self.robot.set_speed_scale(round(float(np.clip(self.robot.speed_scale + step, 0.1, 1.0)), 2))
                logger.info("speed scale: %.0f %%", 100.0 * self.robot.speed_scale)

    def _on_start(self) -> None:
        if self.mode is Mode.FAULT:
            try:
                self.robot.clear_fault()
            except RobotError as exc:
                logger.error("cannot clear the fault: %s", exc)
                return
        elif self.mode is not Mode.IDLE:
            return
        try:
            self.robot.start_streaming()
        except RobotError as exc:
            logger.error("%s", exc)
            return
        self._q_ref = self.robot.commanded_position()
        self._clutch_armed = False
        self._lag_since = None
        self.fault_reason = None
        state = None
        try:
            state = self.robot.get_state()
        except RobotError:
            pass
        measured = None if state is None else state.gripper
        self._gripper = float(self.robot.gripper_command if measured is None else measured)
        self._gripper_desired = self._gripper
        # With a hand closure controlling the gripper, the hand must first match the gripper's
        # present opening, so that starting never opens a gripper that is holding a part.
        self._gripper_needs_sync = self._hand_mode and self.cfg.xr.gripper_source == "driving_hand_fist"
        self._sync_range = None
        self._toggle_closed = self._gripper > 0.5
        self.mode = Mode.READY
        logger.info("READY: hold the clutch (grip button) to move the arm")
        if self.cfg.control.home_on_start:
            if float(np.max(np.abs(self._q_ref - self.model.home))) > math.radians(0.5):
                self._on_home()
            else:
                logger.info("the arm is already at the home pose")

    def _on_gripper_key(self, closing: float) -> None:
        """Open ('o') or close ('g') the gripper now, whatever its reported state.

        The key overrides the hand until the hand takes over again: when it matches the gripper or
        has clearly been opened or closed on purpose. So an open hand does not immediately undo 'g'.
        """
        name = "CLOSE" if closing >= 0.5 else "OPEN"
        if self.mode not in (Mode.READY, Mode.TRACKING, Mode.HOMING):
            logger.info("press 'r' first, then '%s' %ss the gripper", "g" if closing >= 0.5 else "o", name.lower())
            return
        self._gripper = closing
        self._gripper_desired = closing
        self._toggle_closed = closing >= 0.5
        self._gripper_needs_sync = True
        self._sync_range = None
        self.robot.set_gripper(closing)
        self.robot.resend_gripper()
        logger.info("GRIPPER %s requested; your hand takes over once it matches or is moved clearly", name)

    def _on_stop(self) -> None:
        """Stop the arm at once and stop commanding it. The program keeps running; 'r' resumes."""
        if self.mode in (Mode.IDLE, Mode.FAULT):
            return
        self.mapper.release()
        self._home_move = None
        self._clutch_armed = False
        try:
            self.robot.halt()
        except Exception as exc:  # noqa: BLE001 - stopping must never raise
            logger.warning("while stopping the robot: %s", exc)
        self.mode = Mode.IDLE
        logger.warning("STOPPED: the arm is holding where it is. Press 'r' to resume.")

    def _on_home(self) -> None:
        if self.mode not in (Mode.READY, Mode.TRACKING):
            if self.mode is Mode.IDLE:
                logger.info("press 'r' to start before homing")
            return
        self.mapper.release()
        self._home_move = JointMove(self._q_ref, self.model.home, math.radians(self.cfg.control.home_speed_deg_s))
        self._clutch_armed = False
        self.mode = Mode.HOMING
        logger.info("HOMING (%.1f s); squeeze the clutch to cancel", self._home_move.duration)

    def _on_reset(self) -> None:
        if self.mode is Mode.FAULT:
            return
        self.mapper.release()
        try:
            supported = self.robot.reset_scene()
        except RobotError as exc:
            self._enter_fault(str(exc))
            return
        if not supported:
            logger.info("scene reset is only available in simulation")
            return
        self._gripper = float(self.robot.gripper_command)
        self._gripper_desired = self._gripper
        self._toggle_closed = self._gripper > 0.5
        # A trigger already held during reset is not a new toggle press. It must
        # be released before the next press can close the restored gripper.
        self._prev_grip_high = (
            self.last_input.valid and self.last_input.grip >= self.cfg.gripper.toggle_threshold
        )
        if self.mode is not Mode.IDLE:
            self._q_ref = self.robot.commanded_position()
            self._home_move = None
            self._clutch_armed = False
            self._lag_since = None
            self.mode = Mode.READY

    def _on_record(self) -> None:
        if self.recorder is None:
            logger.info("recording is disabled (start with --record)")
        elif self.recorder.active:
            self.recorder.stop()
        elif self.mode in (Mode.IDLE, Mode.FAULT):
            logger.info("press 'r' to start before recording")
        else:
            self.recorder.start()

    # ------------------------------------------------------------------ supervision
    def _enter_fault(self, reason: str) -> None:
        self.mapper.release()
        self._home_move = None
        self.fault_reason = reason
        self.mode = Mode.FAULT
        try:
            self.robot.stop_streaming()
        except Exception as exc:  # noqa: BLE001
            logger.warning("while stopping the robot: %s", exc)
        logger.error("FAULT: %s. The arm is stopped; press 'r' to recover.", reason)

    def _supervise(self, now: float) -> Optional[RobotState]:
        """Check the robot. Returns its state, or ``None`` after switching to FAULT."""
        control = self.cfg.control
        try:
            state = self.robot.get_state()
        except RobotError as exc:
            self._enter_fault(str(exc))
            return None
        self.last_state = state
        if state.fault:
            self._enter_fault(state.fault)
            return None
        if state.age > control.state_timeout_s:
            self._enter_fault(f"no robot feedback for {state.age:.1f} s")
            return None
        if not self.robot.streaming:
            self._enter_fault("set-point streaming stopped")
            return None
        if self.robot.lag_hold_time > control.stall_timeout_s:
            self._enter_fault(
                f"the arm has not followed its command for {self.robot.lag_hold_time:.1f} s "
                "(blocked, stopped or the link to the robot is too slow)." + self.robot.stall_hint()
            )
            return None
        lag = float(np.max(np.abs(self.robot.commanded_position() - state.q)))
        if lag > math.radians(control.tracking_error_limit_deg):
            if self._lag_since is None:
                self._lag_since = now
            elif now - self._lag_since > 0.3:
                self._enter_fault(
                    f"the arm is {math.degrees(lag):.0f} deg away from its command (blocked, stopped or disabled?)."
                    + self.robot.stall_hint()
                )
                return None
        else:
            self._lag_since = None
        return state

    # ------------------------------------------------------------------ modes
    def _clutch_closed(self, inp: OperatorInput) -> bool:
        return inp.valid and inp.hand_pose is not None and inp.clutch >= self.cfg.xr.clutch_on

    def _clutch_open(self, inp: OperatorInput) -> bool:
        # Only a valid reading counts: a controller that dropped out has not "released" anything.
        return inp.valid and inp.clutch <= self.cfg.xr.clutch_off

    def _yaw_offset(self) -> float:
        value = self.cfg.mapping.yaw_offset_deg
        if isinstance(value, str):  # "auto"
            yaw = self.robot.view_yaw()
            return 0.0 if yaw is None else float(yaw)
        return math.radians(float(value))

    def _step_ready(self, inp: OperatorInput) -> None:
        if not self._clutch_armed:
            # The clutch must be seen open once, so holding it while starting never moves the arm.
            self._clutch_armed = self._clutch_open(inp)
            return
        if self._clutch_closed(inp):
            tcp = self.model.fk(self._q_ref)
            error = self.mapper.alignment_error(inp.hand_pose, inp.head_pose, tcp, self._yaw_offset())
            if error > self.cfg.mapping.engage_tolerance_deg:
                # Absolute orientation: engaging now would swing the gripper to the hand's orientation.
                now = time.monotonic()
                if now - self._last_align_log > 1.0:
                    self._last_align_log = now
                    logger.warning(
                        "not engaged: your hand and the gripper differ by %.0f deg. Hold the hand with the "
                        "palm facing down and the fingers pointing forward (gripper down), then squeeze again.",
                        error,
                    )
                return
            self.mapper.engage(inp.hand_pose, inp.head_pose, tcp, self._yaw_offset())
            self._tcp_at_engage = tcp[:3, 3].copy()
            self.mode = Mode.TRACKING
            logger.info("TRACKING")

    def _step_tracking(self, inp: OperatorInput, dt: float) -> None:
        if not inp.valid or inp.hand_pose is None:
            self._release("hand tracking lost: release and squeeze the clutch again", rearm=False)
            return
        if inp.clutch <= self.cfg.xr.clutch_off:
            self._release("READY (clutch open)", rearm=True)
            return
        result = self.mapper.update(inp.hand_pose, dt)
        if result.jump:
            self._release("hand pose jumped: release and squeeze the clutch again", rearm=False)
            return
        solution = self.ik.solve(result.target, self._q_ref)
        self._q_ref = solution.q
        self.robot.set_joint_target(solution.q)
        self.last_ik_error = (solution.position_error, solution.rotation_error)
        self.workspace_limited = result.limited
        if not solution.success and time.monotonic() - self._ik_fail_logged > 2.0:
            self._ik_fail_logged = time.monotonic()
            logger.warning("IK did not converge (%s); holding the best feasible posture", solution.message)

    def _release(self, message: str, rearm: bool) -> None:
        self.mapper.release()
        self._clutch_armed = rearm
        self.mode = Mode.READY
        (logger.info if rearm else logger.warning)("%s", message)

    def _step_homing(self, inp: OperatorInput, dt: float) -> None:
        if not self._clutch_armed:
            self._clutch_armed = self._clutch_open(inp)
        elif self._clutch_closed(inp):
            self._home_move = None
            self._clutch_armed = False
            self.mode = Mode.READY
            logger.info("homing cancelled; %s to take over",
                        "open your left hand, then make a fist" if self._hand_mode
                        else "release and squeeze the clutch")
            return
        self._q_ref = self._home_move.step(dt)
        self.robot.set_joint_target(self._q_ref)
        if self._home_move.done:
            self._home_move = None
            self.mode = Mode.READY
            logger.info("home reached. READY")

    def _update_gripper(self, inp: OperatorInput, dt: float) -> None:
        gripper = self.cfg.gripper
        if not gripper.enabled:
            return
        mode = gripper.mode
        if self._hand_mode and self.cfg.xr.gripper_source in ("other_hand_pinch", "both_hands_pinch"):
            mode = "toggle"  # a pinch is a momentary gesture: each new pinch flips open / closed
        if inp.valid:  # keep the last command if the controller drops out (never drop the part)
            if mode == "analog":
                desired = float(np.clip(inp.grip, 0.0, 1.0))
                if self._gripper_needs_sync:
                    low, high = self._sync_range or (desired, desired)
                    self._sync_range = (min(low, desired), max(high, desired))
                    moved = self._sync_range[1] - self._sync_range[0]
                    # The hand takes over once it matches the gripper, or once it has clearly been
                    # opened or closed on purpose (the gripper's reported opening can be unreliable).
                    if abs(desired - self._gripper) <= 0.2 or moved >= 0.4:
                        self._gripper_needs_sync = False
                        logger.info("gripper: it now follows your hand")
                    else:
                        desired = self._gripper  # keep the present opening
                self._gripper_desired = desired
            else:
                high = inp.grip >= gripper.toggle_threshold
                if high and not self._prev_grip_high:
                    self._toggle_closed = not self._toggle_closed
                    logger.info("GRIPPER %s", "CLOSING" if self._toggle_closed else "OPENING")
                self._prev_grip_high = high
                self._gripper_desired = 1.0 if self._toggle_closed else 0.0
        step = gripper.max_rate * dt
        self._gripper += float(np.clip(self._gripper_desired - self._gripper, -step, step))
        self.robot.set_gripper(self._gripper)

    # ------------------------------------------------------------------ outputs
    def _record(self, now: float, state: RobotState) -> None:
        recorder = self.recorder
        if recorder is None or not recorder.due(now):
            return
        target = self.mapper.target if self.mode is Mode.TRACKING else None
        sample = {
            "mode": self.mode.value,
            "q": state.q,
            "dq": state.dq,
            "q_command": self.robot.commanded_position(),
            "q_target": self._q_ref,
            "tcp_pose": pose_to_xyzquat(self.model.fk(state.q)),
            "tcp_target": pose_to_xyzquat(self.model.fk(self._q_ref) if target is None else target),
            "gripper": state.gripper,
            "gripper_command": self._gripper,
        }
        if "objects" in state.info:
            sample["objects"] = state.info["objects"]
        image = None if self.images is None else self.images.latest(max_age=0.5)
        recorder.add(sample, image)

    def _update_display(self, now: float) -> None:
        if self.images is not None and self.source.wants_images:
            frame, count = self.images.latest_with_count()
            if frame is not None and count != self._last_frame:
                self._last_frame = count
                self.source.show_image(frame)
        if self.viewer is not None and now - self._last_viewer >= 1.0 / 30.0 and self.last_state is not None:
            self._last_viewer = now
            target = self.mapper.target if self.mode is Mode.TRACKING else None
            self.viewer.update(self.last_state.q, self.last_state.gripper, target)

    def _log_status(self, now: float) -> None:
        if now - self._last_status < 2.0:
            return
        self._last_status = now
        if self.mode is Mode.TRACKING:
            arm_travel = float("nan")
            if self.last_state is not None and getattr(self, "_tcp_at_engage", None) is not None:
                arm_travel = float(np.linalg.norm(self.model.fk(self.last_state.q)[:3, 3] - self._tcp_at_engage))
            logger.info(
                "tracking | moved since clutch: hand %.0f mm -> target %.0f mm -> arm %.0f mm | "
                "IK error %.1f mm %.1f deg%s | speed %.0f %% | servo send avg %.1f / max %.0f ms, "
                "loop lateness max %.0f ms | arm lag slowed %.0f %%, held %.0f %% of ticks%s%s | %s",
                1e3 * self.mapper.hand_travel,
                1e3 * self.mapper.target_travel,
                1e3 * arm_travel,
                1e3 * self.last_ik_error[0],
                math.degrees(self.last_ik_error[1]),
                " | at workspace limit" if self.workspace_limited else "",
                100.0 * self.robot.speed_scale,
                1e3 * self.robot.send_avg,
                1e3 * self.robot.send_max,
                1e3 * self.robot.max_lateness,
                *self.robot.lead_stats(),
                (
                    f" | gripper call max {1e3 * self.robot.gripper_call_max:.0f} ms"
                    if getattr(self.robot, "gripper_call_max", 0.0) > 0.0 else ""
                )
                + (" | ARM PAUSED while the gripper moves" if self.robot.paused else "")
                + (f" | arm pauses for the gripper: {self.robot.pauses}" if self.robot.pauses else "")
                + (
                    f" | LATENCY GUARD waited {self.robot.drain_ticks * self.robot.servo_period:.1f} s "
                    "for the controller queue" if self.robot.drain_ticks else ""
                ),
                (
                    " | GRIPPER LOCKED: open or close your right hand to take control"
                    if self._gripper_needs_sync else ""
                ),
                self.robot.gripper_status() or "gripper: -",
            )
        elif self.mode is Mode.READY and not self.last_input.connected:
            logger.info("ready | waiting for the headset (open the web page and press 'Enter VR')")
        elif self.mode is Mode.READY:
            inp = self.last_input
            xr = self.cfg.xr
            clutch_name = f"{xr.hand} grip button"
            if not inp.valid:
                advice = (
                    "hand/controller pose is not arriving or is stale: keep it in view of the headset "
                    f"[{getattr(self.source, 'diagnostics', lambda: '')()}]"
                )
            elif not self._clutch_armed:
                advice = (
                    "open your left hand completely first" if self._hand_mode
                    else f"release the {clutch_name} completely first"
                )
            else:
                advice = (
                    "make a fist with your left hand to move the arm (right hand palm down, fingers forward)"
                    if self._hand_mode
                    else f"squeeze and hold the {clutch_name} to move the arm"
                )
            
            logger.info(
                "ready | clutch %.2f (engages at >= %.2f)%s | gripper input %.2f | pose valid: %s | armed: %s | %s -> %s",
                inp.clutch, xr.clutch_on, f" [{inp.detail}]" if inp.detail else "", inp.grip,
                "yes" if inp.valid else "NO", "yes" if self._clutch_armed else "no",
                self.robot.gripper_status() or "gripper: -", advice,
            )
