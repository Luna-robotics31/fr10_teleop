"""The gripper in a process of its own, shaped like the Inspire hand controller of xr_teleoperate.

In the original project the arm and the hands never wait for each other: the hand controller is a
separate ``Process`` that reads the target from shared memory and publishes at its own fixed rate,
while the arm controller streams from its own thread. The FR10 gripper is different in one way: its
commands (``ActGripper`` / ``MoveGripper``) are request/response calls to the same controller that
carries the arm's joint commands, so the same shape is used with extra care:

* a separate process (nothing in it can hold up the arm's 125 Hz loop, and nothing there can be
  held up by it), connected to the controller with plain XML-RPC and a timeout of its own;
* the teleoperation only writes the wanted closure into a shared value (``set_closure``), exactly
  as the original writes the hand pose into a shared ``Array``;
* a fixed 20 Hz loop that sends at most one command every ``min_interval_s`` and only when the
  position changed, followed by one exact final command when the fingers stop;
* the gripper is activated once, right after the program connects and before the arm streams, because
  activating takes seconds and must not overlap the arm's commands.

Measured on the FR10 controller (software V3.9.4): ``MoveGripper`` runs in the same motion queue as the
arm's ``ServoJ`` set-points. While a gripper move runs (~1.2 s for a full stroke) the arm stands still and
the set-points streamed meanwhile pile up behind it (``mc_queue_len`` grows to ~160), so the arm then lags
~1.3 s for the rest of the session, and every later ``MoveGripper`` waits for a queue that never empties
(they all run when the teleoperation ends). UDP or XML-RPC, ``ServoMoveStart``, ``block`` 0 or 1 make no
difference. So, with ``fairino.gripper.pause_arm_for_moves`` (the default), each gripper move while the arm
streams is a handshake with the arm's servo thread: the process sets ``pause_request``, the arm brakes
smoothly to rest and stops sending (``arm_paused``), the process sends the move (taken at once: the queue
is empty), withdraws the request, and the arm stays silent until the state packet reports the gripper move
done, then follows the hand again with an empty queue. To keep those pauses rare, the proportional mode
sends a position only once the fingers have settled (or at fully open / closed).

Depends on the standard library, NumPy and the modules of this package that need no robot SDK.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from typing import Callable, Optional

from .config import Config
from .pinch_gripper import ProportionalPinchController, ScriptGripper
from .robots.base import RobotError

logger = logging.getLogger(__name__)

LOOP_PERIOD_S = 0.05  # the process polls the shared target at 20 Hz

# layout of the shared status array
HEARTBEAT, POSITION, FAILURES, BUSY_SINCE, COMMANDS, LAST_OK, STATE, ARM_PAUSED_FOR, LAST_PAUSE_S, STATUS_LEN = range(10)
STATE_STARTING, STATE_ACTIVATING, STATE_RUNNING, STATE_FAILED = 0.0, 1.0, 2.0, 3.0

ARM_BRAKE_TIMEOUT_S = 2.5  # longest wait for the arm to come to rest after asking it to pause


class _PausingGripper:
    """Sends each move while the arm's set-point stream is paused (see the module docstring)."""

    def __init__(self, gripper, streaming, pause_request, arm_paused, status, brake_timeout=ARM_BRAKE_TIMEOUT_S):
        self._gripper = gripper
        self._streaming, self._pause_request, self._arm_paused = streaming, pause_request, arm_paused
        self._status = status
        self._brake_timeout = brake_timeout

    def __getattr__(self, name):  # activate(), close_connection(), activate_each_move, block, ...
        return getattr(self._gripper, name)

    def move_percent(self, percent, **kwargs):
        if not self._streaming.value:
            return self._gripper.move_percent(percent, **kwargs)  # the arm is not streaming: no pause needed
        asked = time.monotonic()
        self._pause_request.set()
        try:
            while not self._arm_paused.is_set():
                if not self._streaming.value:
                    break  # streaming ended meanwhile: the controller is quiet anyway
                if time.monotonic() - asked > self._brake_timeout:
                    raise RobotError(f"the arm did not pause for the gripper within {self._brake_timeout:.1f} s")
                self._arm_paused.wait(0.01)
            self._status[ARM_PAUSED_FOR] = time.monotonic()
            self._gripper.move_percent(percent, **kwargs)  # taken at once: the controller's queue is empty
        finally:
            self._pause_request.clear()  # the arm follows the hand again
            if self._status[ARM_PAUSED_FOR]:
                self._status[LAST_PAUSE_S] = time.monotonic() - asked
            self._status[ARM_PAUSED_FOR] = 0.0


class _TrackedGripper:
    """Wraps a gripper so the process can publish what it is doing (busy, last command, failures)."""

    def __init__(self, gripper, status):
        self._gripper = gripper
        self._status = status

    def move_percent(self, percent, **kwargs):
        self._status[BUSY_SINCE] = time.monotonic()
        try:
            self._gripper.move_percent(percent, **kwargs)
        except Exception:
            self._status[FAILURES] += 1
            raise
        finally:
            self._status[BUSY_SINCE] = 0.0
        self._status[POSITION] = float(round(percent))
        self._status[COMMANDS] += 1
        self._status[LAST_OK] = time.monotonic()
        self._status[FAILURES] = 0
        if round(percent) in (0, 100):
            logger.info("gripper moved to %d %%", round(percent))


def gripper_main(
    cfg: Config,
    target,
    status,
    resend,
    stop,
    ready,
    gripper_factory: Optional[Callable[[Config], object]] = None,
    max_iterations: Optional[int] = None,
    streaming=None,
    pause_request=None,
    arm_paused=None,
) -> None:
    """Entry point of the gripper process (also runs in a thread in the tests).

    ``streaming`` (shared value, true while the arm streams), ``pause_request`` and ``arm_paused``
    (events) are the pause handshake with the arm; without them moves are sent straight away.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    gc = cfg.fairino.gripper
    status[STATE] = STATE_STARTING
    try:
        gripper = (gripper_factory or ScriptGripper.connect_plain)(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("the gripper process cannot start: %s", exc)
        status[STATE] = STATE_FAILED
        ready.set()
        return
    status[STATE] = STATE_ACTIVATING
    logger.info("activating gripper %d (this can take several seconds; the arm waits for it) ...", gc.index)
    try:
        gripper.activate()
    except RobotError as exc:
        logger.warning("gripper activation did not complete (%s); continuing, it may already be active", exc)
    gripper.activate_each_move = gc.activate_before_move
    gripper.block = 0 if gc.wait_for_move else 1
    pausing = gc.pause_arm_for_moves and None not in (streaming, pause_request, arm_paused)
    if pausing:
        logger.info(
            "while the arm streams, the arm pauses for each gripper move until the gripper has moved "
            "(~1 s; the controller cannot move both at once)"
        )
        sender = _PausingGripper(gripper, streaming, pause_request, arm_paused, status)
        follower = ProportionalPinchController(
            _TrackedGripper(sender, status),
            resolution=gc.resolution_percent,
            min_interval_s=max(gc.min_interval_s, gc.pause_cooldown_s),
            settled_only=True,  # one move per gesture: every move costs an arm pause
            still_band=gc.still_band_percent,
        )
    else:
        follower = ProportionalPinchController(
            _TrackedGripper(gripper, status), resolution=gc.resolution_percent, min_interval_s=gc.min_interval_s
        )
    status[STATE] = STATE_RUNNING
    ready.set()

    open_close = gc.mode == "open_close"
    closed = target.value >= 0.5
    iterations = 0
    while not stop.is_set():
        if resend.is_set():
            resend.clear()
            follower.resend()
        closure = float(target.value)
        if open_close:  # only open or closed, with a gap between the two thresholds
            closed = closure >= gc.close_above or (closed and closure > gc.open_below)
            closure = 1.0 if closed else 0.0
        follower.update_closure(closure, time.monotonic())
        status[HEARTBEAT] = time.monotonic()
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        stop.wait(LOOP_PERIOD_S)
    try:
        gripper.close_connection()
    except Exception:  # noqa: BLE001
        pass


class GripperProcess:
    """Handle used by the teleoperation: ``set_closure`` writes the target, the process does the rest."""

    BUSY_REPORT_S = 3.0  # a move is normally taken in milliseconds (the arm is paused, the queue empty)

    def __init__(self, cfg: Config, initial_closure: float = 0.0):
        context = multiprocessing.get_context("spawn")  # a clean interpreter: no forked threads or sockets
        self._cfg = cfg
        self._target = context.Value("d", float(initial_closure), lock=False)
        self._status = context.Array("d", STATUS_LEN, lock=False)
        self._resend = context.Event()
        self._stop = context.Event()
        self._ready = context.Event()
        # the pause handshake with the arm's servo thread
        self._streaming = context.Value("b", 0, lock=False)
        self._pause_request = context.Event()
        self._arm_paused = context.Event()
        self._process = context.Process(
            target=gripper_main,
            args=(cfg, self._target, self._status, self._resend, self._stop, self._ready),
            kwargs=dict(streaming=self._streaming, pause_request=self._pause_request, arm_paused=self._arm_paused),
            name="fr10-gripper",
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()

    def set_closure(self, closure: float) -> None:
        self._target.value = float(closure)

    def resend(self) -> None:
        self._resend.set()

    def wait_ready(self, timeout: float) -> bool:
        """Wait until the gripper has been activated (or activation was given up)."""
        return self._ready.wait(timeout)

    # ---- pause handshake (called by the arm's backend)
    def set_streaming(self, streaming: bool) -> None:
        self._streaming.value = 1 if streaming else 0
        if not streaming:
            self._arm_paused.clear()

    def pause_requested(self) -> bool:
        """Checked by the arm's servo thread every tick."""
        return self._pause_request.is_set() and self._process.is_alive()

    def set_arm_paused(self, paused: bool) -> None:
        if paused:
            self._arm_paused.set()
        else:
            self._arm_paused.clear()

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def status_text(self) -> str:
        if not self._process.is_alive():
            return "GRIPPER PROCESS NOT RUNNING"
        state = self._status[STATE]
        if state == STATE_FAILED:
            return "GRIPPER PROCESS FAILED TO START (cannot reach the controller?)"
        if state in (STATE_STARTING, STATE_ACTIVATING):
            return "gripper process: activating the gripper"
        busy = self._status[BUSY_SINCE]
        if busy and time.monotonic() - busy > self.BUSY_REPORT_S:
            return (
                f"gripper process BLOCKED in a controller call for {time.monotonic() - busy:.0f} s "
                "(the controller is not answering the gripper command)"
            )
        if self._status[ARM_PAUSED_FOR]:
            return "gripper move being sent: arm paused (the controller cannot move both at once)"
        position = self._status[POSITION]
        shown = "no command yet" if self._status[COMMANDS] == 0 else f"at {position:.0f} %"
        failures = int(self._status[FAILURES])
        last_pause = self._status[LAST_PAUSE_S]
        return f"gripper process running ({shown}, {int(self._status[COMMANDS])} sent" + (
            f", last arm pause {last_pause:.1f} s" if last_pause else ""
        ) + (f", {failures} FAILED" if failures else "") + ")"

    def stop(self) -> None:
        self._stop.set()
        if self._process.is_alive() or self._process.pid is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
