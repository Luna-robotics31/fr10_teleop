"""Robot backend for the real FAIRINO FR10 (vendor Python SDK, ``Robot.RPC``).

Motion uses the controller's joint servo mode: ``ServoMoveStart`` then one ``ServoJ`` every
servo period (8 ms by default, the period FAIRINO's own ROS 2 driver uses) and finally
``ServoMoveEnd``. Joint angles in the URDF and on the controller are identical apart from
the unit (radians here, degrees on the controller).

Feedback comes from the SDK's real-time state packet (TCP port 20004). The gripper is driven
through the controller (``ActGripper`` / ``MoveGripper``) from a separate low-rate thread.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import threading
import time
import xmlrpc.client
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..config import Config
from .base import Robot, RobotError, RobotState

logger = logging.getLogger(__name__)

# Meaning of the SDK return codes we can explain (everything else: FAIRINO error-code manual).
_ERROR_HINTS = {
    -4: "the SDK is not connected to the controller",
    -2: "UDP transmission failed",
    -1: "command could not be sent",
    99: "a safety stop is active",
}


def describe_error(code: int) -> str:
    hint = _ERROR_HINTS.get(int(code))
    return f"error code {code}" + (f" ({hint})" if hint else "")


def load_fairino_sdk(sdk_dir: Path):
    """Import the vendor ``Robot`` module (``Robot.py`` or ``Robot.*.so``) from ``sdk_dir``."""
    sdk_dir = Path(sdk_dir).expanduser().resolve()
    if not sdk_dir.is_dir() or not (list(sdk_dir.glob("Robot.py")) or list(sdk_dir.glob("Robot.*.so"))):
        raise RobotError(
            f"FAIRINO SDK not found in {sdk_dir}. Set 'fairino.sdk_path' to the folder that contains "
            "Robot.py (the 'fairino' folder of the vendor's Python SDK)."
        )
    loaded = sys.modules.get("Robot")
    if loaded is not None:
        origin = Path(getattr(loaded, "__file__", "") or "").resolve().parent
        if origin == sdk_dir and hasattr(loaded, "RPC"):
            return loaded
        raise RobotError(f"another module named 'Robot' is already imported from {origin}")
    sys.path.insert(0, str(sdk_dir))
    try:
        module = importlib.import_module("Robot")
    except Exception as exc:  # noqa: BLE001 - vendor code, anything can go wrong
        sys.modules.pop("Robot", None)
        raise RobotError(f"cannot import the FAIRINO SDK from {sdk_dir}: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(sdk_dir))
        except ValueError:
            pass
    if not hasattr(module, "RPC"):
        raise RobotError(f"{module.__file__} is not a FAIRINO SDK (no RPC class)")
    return module


def _parameter_names(func: Callable) -> Optional[List[str]]:
    """Parameter names of an SDK method, or ``None`` if it cannot be inspected (compiled SDK)."""
    try:
        return [name for name in inspect.signature(func).parameters if name != "self"]
    except (TypeError, ValueError):
        return None


class _TimeoutTransport(xmlrpc.client.Transport):
    def __init__(self, timeout: float):
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self._timeout
        return connection


class FairinoRobot(Robot):
    name = "fr10"
    RPC_PORT = 20003  # controller XML-RPC command port
    GRIPPER_BUSY_REPORT_S = 1.0  # a gripper call longer than this is reported in the status line

    def __init__(self, cfg: Config, lower: Sequence[float], upper: Sequence[float], sdk=None):
        super().__init__(cfg, lower, upper)
        self.pause_timeout_s = cfg.fairino.gripper.pause_timeout_s
        self.max_backlog_points = cfg.fairino.max_queue_points or None
        self._sdk = sdk
        self._rpc = None
        self._rpc_lock = threading.Lock()  # the SDK's XML-RPC proxy is not thread-safe
        self._servo_j: Optional[Callable[[List[float]], int]] = None
        self._servo_start_kwargs: dict = {}
        self._gripper_args: Optional[Callable[..., tuple]] = None
        self._gripper_thread: Optional[threading.Thread] = None
        self._gripper_process = None  # GripperProcess: the gripper in a process of its own (default)
        self._gripper_resend = threading.Event()  # send the present gripper command again (e.g. the 'o' key)
        self._gripper_wake = threading.Event()  # wakes the gripper thread when the command changes
        self._gripper_last = None  # (label, ok, message, monotonic time) of the latest command tried
        self._gripper_commands = 0
        self._gripper_busy_since = None  # monotonic time the gripper thread entered a controller call
        self._gripper_driver = None  # ScriptGripper on its own SDK connection (the gripper.py calls)
        self._gripper_needs_activation = True
        self.gripper_call_max = 0.0  # longest ActGripper + MoveGripper round trip [s]
        self._gripper_failures = 0
        self._gripper_stop = threading.Event()
        self._last_packet = None
        self._zero_packets = 0
        self._q: Optional[np.ndarray] = None
        self._dq = np.zeros(6)
        self._stamp = 0.0

    # ------------------------------------------------------------------ connection
    def _connect(self) -> None:
        fc = self.cfg.fairino
        sdk = self._sdk if self._sdk is not None else load_fairino_sdk(fc.sdk_dir)
        self._sdk = sdk
        # The SDK keeps "not connected" in a class attribute that survives failed attempts.
        if hasattr(sdk.RPC, "is_connect"):
            sdk.RPC.is_connect = True
        logger.info("[fr10] connecting to the controller at %s ...", fc.ip)
        try:
            self._rpc = sdk.RPC(fc.ip)
        except Exception as exc:  # noqa: BLE001
            raise RobotError(f"FAIRINO SDK failed to connect to {fc.ip}: {exc}") from exc
        try:
            if getattr(sdk.RPC, "is_connect", True) is False:
                raise RobotError(
                    f"no answer from the controller at {fc.ip}. Check the Ethernet cable, that this PC has "
                    "an address in the same subnet (e.g. 192.168.58.10) and that the controller is on."
                )
            self._wait_for_feedback(3.0)
            self._log_versions()
            self._bind_sdk_calls()
            self._prepare_robot()
            self._prepare_gripper()
        except Exception:
            self._disconnect()
            raise

    def _disconnect(self) -> None:
        self._gripper_stop.set()
        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=1.0)
            self._gripper_thread = None
        driver, self._gripper_driver = self._gripper_driver, None
        if driver is not None:
            driver.close_connection()
        process, self._gripper_process = self._gripper_process, None
        if process is not None:
            process.stop()
        rpc, self._rpc = self._rpc, None
        if rpc is not None:
            try:
                rpc.CloseRPC()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[fr10] CloseRPC: %s", exc)

    def _packet(self):
        """Current real-time state packet, or ``None`` before the first one has arrived."""
        packet = getattr(self._rpc, "robot_state_pkg", None)
        if packet is None or isinstance(packet, type):  # the SDK stores the class itself until then
            return None
        return packet

    def _wait_for_feedback(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._read_state()
                return
            except RobotError:
                time.sleep(0.05)
        raise RobotError(
            "connected, but the controller sends no joint feedback (real-time port 20004). "
            "Is another program already connected to the robot?"
        )

    def _log_versions(self) -> None:
        for label, getter in (("SDK", "GetSDKVersion"), ("controller software", "GetSoftwareVersion")):
            try:
                with self._rpc_lock:
                    result = getattr(self._rpc, getter)()
                logger.info("[fr10] %s: %s", label, result[1:] if isinstance(result, (list, tuple)) else result)
            except Exception as exc:  # noqa: BLE001 - informative only
                logger.debug("[fr10] %s failed: %s", getter, exc)

    def _bind_sdk_calls(self) -> None:
        """Adapt to the SDK version in use (the ServoJ / MoveGripper signatures changed over time)."""
        fc = self.cfg.fairino
        rpc = self._rpc
        use_udp = fc.servo_transport == "udp"

        servo_params = _parameter_names(rpc.ServoJ)
        kwargs: dict = {"cmdT": self.servo_period}
        if servo_params is None or "axisPos" in servo_params:
            kwargs["axisPos"] = [0.0, 0.0, 0.0, 0.0]  # no external axes
        if use_udp:
            if servo_params is not None and "cmdType" not in servo_params:
                raise RobotError("this SDK version has no UDP ServoJ; set fairino.servo_transport: xmlrpc")
            kwargs["cmdType"] = 1
        self._servo_j = lambda q_deg: rpc.ServoJ(joint_pos=q_deg, **kwargs)
        logger.info(
            "[fr10] arm set-points (ServoJ) go over %s every %.0f ms",
            "UDP (fire and forget)" if use_udp else "XML-RPC (waits for each reply)", 1e3 * self.servo_period,
        )

        self._servo_start_kwargs = {}
        if use_udp and hasattr(rpc, "ServoMoveStart"):
            start_params = _parameter_names(rpc.ServoMoveStart)
            if start_params is None or "cmdType" in start_params:
                self._servo_start_kwargs = {"cmdType": 1}

        gc = fc.gripper
        gripper_params = _parameter_names(rpc.MoveGripper) if hasattr(rpc, "MoveGripper") else None
        short_signature = gripper_params is not None and len(gripper_params) <= 6

        def gripper_args(position: int, block: int = 1) -> tuple:
            """Arguments of ``MoveGripper``; ``block``: 0 = wait for the move, 1 = return at once."""
            args = (
                int(gc.index), int(position), int(gc.speed_percent), int(gc.force_percent),
                int(gc.max_time_ms), int(block),
            )
            # Full signature: (index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque).
            # The controller's XML-RPC server is strict about types: rotNum is a DOUBLE, the rest are
            # integers (the vendor SDK converts them, but the gripper thread bypasses the SDK).
            return args if short_signature else args + (0, 0.0, 0, 0)

        self._gripper_args = gripper_args

    def _rpc_call(self, method: str, *args, timeout: Optional[float] = None) -> int:
        """One-off controller command on a private connection with a timeout.

        The SDK's own wrappers retry for ever when the controller does not answer; set-up
        commands must fail loudly instead.
        """
        fc = self.cfg.fairino
        proxy = xmlrpc.client.ServerProxy(
            f"http://{fc.ip}:{fc.rpc_port}", transport=_TimeoutTransport(fc.rpc_timeout_s if timeout is None else timeout)
        )
        try:
            result = getattr(proxy, method)(*args)
        except Exception as exc:  # noqa: BLE001
            raise RobotError(f"{method}{args} got no answer from the controller: {exc}") from exc
        return int(result[0] if isinstance(result, (list, tuple)) else result)

    def _wait_until(self, condition: Callable[[], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.05)
        return condition()

    def _error_codes(self) -> tuple:
        packet = self._packet()
        return (0, 0) if packet is None else (int(packet.main_code), int(packet.sub_code))

    def _prepare_robot(self) -> None:
        fc = self.cfg.fairino
        packet = self._packet()
        if int(packet.EmergencyStop):
            raise RobotError("the emergency stop is pressed. Release it and clear the error on the pendant.")
        if int(packet.robot_state) == 4:
            raise RobotError("the robot is in drag-teach mode. Leave it on the pendant first.")
        if any(self._error_codes()):
            main, sub = self._error_codes()
            if not fc.reset_errors_on_connect:
                raise RobotError(
                    f"the controller reports error {main}/{sub}. Clear it on the pendant, or start with "
                    "--reset-errors to let this program call ResetAllError()."
                )
            logger.warning("[fr10] controller error %d/%d active, calling ResetAllError()", main, sub)
            self._rpc_call("ResetAllError")
            if not self._wait_until(lambda: not any(self._error_codes()), 3.0):
                main, sub = self._error_codes()
                raise RobotError(f"error {main}/{sub} is still active after ResetAllError(); check the pendant.")
        if int(self._packet().robot_mode) != 0:
            if not fc.switch_to_auto_mode:
                raise RobotError("the robot is in manual mode. Switch it to automatic on the pendant.")
            logger.info("[fr10] switching the robot to automatic mode")
            code = self._rpc_call("Mode", 0)
            if code != 0 or not self._wait_until(lambda: int(self._packet().robot_mode) == 0, 5.0):
                raise RobotError(
                    f"could not switch to automatic mode ({describe_error(code)}). Switch it on the pendant."
                )
        if fc.enable_on_connect:
            code = self._rpc_call("RobotEnable", 1)
            if code != 0:
                raise RobotError(f"RobotEnable(1) failed: {describe_error(code)}")
        logger.info(
            "[fr10] ready: mode=%s enable_state=%s joints=%s deg",
            "automatic" if int(self._packet().robot_mode) == 0 else "manual",
            getattr(self._packet(), "rbtEnableState", "?"),
            np.round(np.rad2deg(self._read_state().q), 2).tolist(),
        )

    def _prepare_gripper(self) -> None:
        gc = self.cfg.fairino.gripper
        if not (gc.enabled and self.cfg.gripper.enabled):
            logger.info("[fr10] gripper disabled")
            return
        if not hasattr(self._rpc, "MoveGripper"):
            raise RobotError("this SDK has no gripper interface; set fairino.gripper.enabled: false")
        if gc.use_process:
            # Like the Inspire hand controller of the original project: a process of its own, fed through a
            # shared value. It activates the gripper itself, before the arm starts streaming.
            measured = self._gripper_fraction(self._packet())
            if measured is not None:
                self.set_gripper(measured)  # start from the present opening
            from ..gripper_process import GripperProcess

            self._gripper_process = GripperProcess(self.cfg, initial_closure=self.gripper_command)
            self._gripper_process.start()
            logger.info("[fr10] the gripper runs in its own process; it activates the gripper now (takes seconds)")
            return

        if gc.pause_arm_for_moves:
            logger.warning(
                "[fr10] the gripper thread (fairino.gripper.use_process: false) cannot pause the arm: while the arm "
                "streams the controller will hold all but the first gripper move. Use the gripper process."
            )

        def active() -> bool:
            packet = self._packet()
            return packet is not None and bool((int(getattr(packet, "gripper_active", 0)) >> gc.index) & 1)

        if not active():
            logger.info("[fr10] activating gripper %d (it may move)", gc.index)
            with self._rpc_lock:
                code = self._rpc.ActGripper(gc.index, 1)
            if code != 0:
                raise RobotError(
                    f"ActGripper({gc.index}, 1) failed: {describe_error(code)}. Check the gripper number on "
                    "the pendant, or set fairino.gripper.enabled: false to run without a gripper."
                )
            if not self._wait_until(active, 5.0):
                logger.warning("[fr10] gripper %d does not report 'active'; continuing", gc.index)
        # Start from the present opening so that connecting never moves the gripper.
        measured = self._gripper_fraction(self._packet())
        if measured is not None:
            self.set_gripper(measured)
        # By default the gripper uses plain XML-RPC calls on their own socket (with a timeout), so a slow
        # gripper command never waits on the arm's connection. A second vendor-SDK connection is optional.
        if gc.separate_sdk_connection:
            try:
                from ..pinch_gripper import ScriptGripper

                self._gripper_driver = ScriptGripper.connect(self.cfg, sdk=self._sdk)
                logger.info("[fr10] gripper uses its own SDK connection")
            except Exception as exc:  # noqa: BLE001 - fall back to plain XML-RPC calls
                self._gripper_driver = None
                logger.warning("[fr10] no separate gripper SDK connection (%s); using plain XML-RPC calls", exc)
        else:
            logger.info("[fr10] gripper uses plain XML-RPC calls with a %.0f s timeout", self.cfg.fairino.rpc_timeout_s)
        self._gripper_stop.clear()
        self._gripper_thread = threading.Thread(target=self._gripper_loop, name="fr10-gripper", daemon=True)
        self._gripper_thread.start()

    # ------------------------------------------------------------------ feedback
    def _gripper_fraction(self, packet) -> Optional[float]:
        gc = self.cfg.fairino.gripper
        if packet is None or not gc.enabled:
            return None
        span = gc.closed_percent - gc.open_percent
        if abs(span) < 1e-6:
            return None
        return float(np.clip((float(packet.gripper_position) - gc.open_percent) / span, 0.0, 1.0))

    def _fault(self, packet) -> Optional[str]:
        rpc = self._rpc
        if getattr(rpc, "reconnect_flag", False) or getattr(rpc, "SDK_state", True) is False:
            return "connection to the controller lost"
        if int(packet.EmergencyStop):
            return "emergency stop pressed"
        main, sub = int(packet.main_code), int(packet.sub_code)
        if main or sub:
            return f"controller error {main}/{sub} (see the FAIRINO error-code manual)"
        if int(packet.safety_stop0_state) or int(packet.safety_stop1_state):
            return "safety stop input active"
        if int(packet.collisionState):
            return "collision detected"
        if int(packet.robot_state) == 4:
            return "robot is in drag-teach mode"
        if int(packet.robot_mode) != 0:
            return "robot is in manual mode"
        return None

    def _read_state(self) -> RobotState:
        if self._rpc is None:
            raise RobotError("fr10: not connected")
        packet = self._packet()
        if packet is not None:
            new_packet = packet is not self._last_packet
            self._last_packet = packet
            q_deg = np.array([packet.jt_cur_pos[i] for i in range(6)], dtype=float)
            valid = bool(np.all(np.isfinite(q_deg)) and np.all(np.abs(q_deg) < 400.0))
            if valid and q_deg[0] == 0.0 and q_deg[1] == 0.0 and q_deg[2] == 0.0:
                # The SDK zeroes j1..j3 of the packet it holds when the next one fails its
                # checksum. Believe exact zeros only once several distinct packets agree.
                if new_packet:
                    self._zero_packets += 1
                valid = self._zero_packets >= 3
            elif valid:
                self._zero_packets = 0
            if valid:
                self._q = np.deg2rad(q_deg)
                self._dq = np.deg2rad(np.array([packet.actual_qd[i] for i in range(6)], dtype=float))
                if new_packet or self._stamp == 0.0:
                    self._stamp = time.monotonic()
        if self._q is None:
            raise RobotError("fr10: no joint feedback received yet")
        return RobotState(
            stamp=self._stamp,
            q=self._q.copy(),
            dq=self._dq.copy(),
            gripper=self._gripper_fraction(packet),
            fault=None if packet is None else self._fault(packet),
        )

    # ------------------------------------------------------------------ motion
    def _on_stream_start(self) -> None:
        process = self._gripper_process
        if process is not None:
            # Activating the gripper takes seconds and uses the same controller: finish it before the arm streams.
            if not process.wait_ready(self.cfg.fairino.gripper.activation_timeout_s + 5.0):
                logger.warning("[fr10] the gripper is still activating; starting the arm anyway")
            # From now on each gripper move pauses the arm. A move the process sent just before this must
            # finish first: set-points streamed while it runs would queue behind it and make the arm lag.
            process.set_streaming(True)
            self._wait_for_quiet_queue()
        if self.cfg.fairino.use_servo_move_start_end and hasattr(self._rpc, "ServoMoveStart"):
            with self._rpc_lock:
                code = self._rpc.ServoMoveStart(**self._servo_start_kwargs)
            if code != 0:
                raise RobotError(f"ServoMoveStart failed: {describe_error(code)}")
        if process is not None:
            process.resend()  # drive the gripper to what the operator's hand shows (pausing the arm)

    def _wait_for_quiet_queue(self) -> None:
        """Wait (up to the gripper's time limit) until no gripper move runs and the motion queue is empty."""
        time.sleep(self.GRIPPER_START_S)  # a move sent a moment ago shows up in the state packet by now
        deadline = time.monotonic() + self.cfg.fairino.gripper.max_time_ms / 1000.0 + 0.5
        while time.monotonic() < deadline:
            packet = self._packet()
            if packet is None or (
                int(getattr(packet, "gripper_motiondone", 1)) != 0 and int(getattr(packet, "mc_queue_len", 0)) == 0
            ):
                return
            time.sleep(0.02)
        logger.warning("[fr10] the controller still reports a gripper move or queued motion; starting anyway")

    def _pause_requested(self) -> bool:
        process = self._gripper_process
        return process is not None and self.cfg.fairino.gripper.pause_arm_for_moves and process.pause_requested()

    def _set_paused(self, paused: bool) -> None:
        process = self._gripper_process
        if process is not None:
            process.set_arm_paused(paused)

    GRIPPER_START_S = 0.1  # a gripper move shows up in the state packet within this time after it was taken

    def _pause_can_end(self, since_withdrawn: float) -> bool:
        """The gripper move runs in the controller's motion queue (and the arm cannot move meanwhile):
        set-points sent before it has finished pile up behind it and the arm then lags by that much for
        the rest of the session. So the stream stays silent until the packet says the move is done."""
        if since_withdrawn < self.GRIPPER_START_S:
            return False
        if since_withdrawn > self.cfg.fairino.gripper.max_time_ms / 1000.0 + 0.5:
            return True  # the gripper's own time limit has passed
        packet = self._packet()
        if packet is None:
            return True
        gripper_moving = int(getattr(packet, "gripper_motiondone", 1)) == 0  # 0 while moving (3 when done)
        return not gripper_moving and int(getattr(packet, "mc_queue_len", 0)) == 0

    def _command_backlog(self) -> int:
        packet = self._packet()
        return 0 if packet is None else int(getattr(packet, "mc_queue_len", 0))

    def _on_stream_stop(self) -> None:
        if self._gripper_process is not None:
            self._gripper_process.set_streaming(False)
        rpc = self._rpc
        if rpc is None or not (self.cfg.fairino.use_servo_move_start_end and hasattr(rpc, "ServoMoveEnd")):
            return
        # The servo thread may be stuck inside the SDK (lost connection): never wait for ever.
        if not self._rpc_lock.acquire(timeout=1.0):
            logger.warning("[fr10] controller link busy, ServoMoveEnd not sent")
            return
        try:
            code = rpc.ServoMoveEnd(**self._servo_start_kwargs)
        finally:
            self._rpc_lock.release()
        if code != 0:
            logger.warning("[fr10] ServoMoveEnd: %s", describe_error(code))

    def _send_setpoint(self, q: Optional[np.ndarray], gripper: float) -> None:
        if q is None:
            return  # holding: the controller keeps the last servo target
        q_deg = [float(v) for v in np.rad2deg(q)]
        with self._rpc_lock:
            code = self._servo_j(q_deg)
        if code != 0:
            raise RobotError(f"ServoJ rejected the set-point: {describe_error(code)}")

    def halt(self) -> None:
        """Stop sending set-points, then ask the controller to stop any motion in progress."""
        self.stop_streaming()
        if self._rpc is None:
            return
        try:
            code = self._rpc_call("StopMotion")
            if code != 0:
                logger.warning("[fr10] StopMotion: %s", describe_error(code))
        except RobotError as exc:  # the arm has already stopped receiving set-points
            logger.warning("[fr10] StopMotion not confirmed: %s", exc)

    def stall_hint(self) -> str:
        if self.cfg.fairino.servo_transport == "udp":
            return (
                " If the arm never moved at all, the controller may not support UDP servo commands: "
                "restart with --set fairino.servo_transport=xmlrpc."
            )
        return ""

    def set_gripper(self, closing: float) -> None:
        super().set_gripper(closing)
        process = self._gripper_process
        if process is not None:
            process.set_closure(self.gripper_command)  # shared memory write: never blocks
        self._gripper_wake.set()  # the gripper thread reacts at once instead of on its next poll

    def resend_gripper(self) -> None:
        """Send the present gripper command again (the gripper thread picks it up at once)."""
        self._gripper_resend.set()
        self._gripper_wake.set()
        if self._gripper_process is not None:
            self._gripper_process.resend()

    def gripper_status(self) -> str:
        if self._gripper_process is not None:
            return self._gripper_process.status_text()
        thread = self._gripper_thread
        if not (self.cfg.fairino.gripper.enabled and self.cfg.gripper.enabled):
            return "gripper disabled"
        if thread is None or not thread.is_alive():
            return "GRIPPER THREAD NOT RUNNING"
        via = "own SDK connection" if self._gripper_driver is not None else "plain XML-RPC"
        busy = self._gripper_busy_since
        if busy is not None and time.monotonic() - busy > self.GRIPPER_BUSY_REPORT_S:
            return (
                f"gripper thread BLOCKED in a controller call for {time.monotonic() - busy:.0f} s ({via}); "
                "the controller is not answering the gripper command"
            )
        if self._gripper_last is None:
            return f"gripper thread running ({via}), no command sent yet"
        label, ok, message, stamp = self._gripper_last
        age = time.monotonic() - stamp
        return (
            f"gripper thread running ({via}); last command {label}: "
            + ("accepted" if ok else f"FAILED ({message})") + f", {age:.0f} s ago; {self._gripper_commands} sent"
        )

    def _clear_fault(self) -> None:
        if any(self._error_codes()):
            logger.warning("[fr10] calling ResetAllError()")
            self._rpc_call("ResetAllError")
            self._wait_until(lambda: not any(self._error_codes()), 3.0)
        packet = self._packet()
        if packet is not None and int(packet.robot_mode) != 0 and self.cfg.fairino.switch_to_auto_mode:
            self._rpc_call("Mode", 0)
            self._wait_until(lambda: int(self._packet().robot_mode) == 0, 5.0)
        if self.cfg.fairino.enable_on_connect:
            self._rpc_call("RobotEnable", 1)

    # ------------------------------------------------------------------ gripper
    def open_gripper(self, wait: bool = True, activate: bool = True) -> None:
        """Open the gripper: ``ActGripper(index, 1)`` then ``MoveGripper(index, open_percent, ...)``."""
        self._command_gripper(self.cfg.fairino.gripper.open_percent, 0.0, wait, activate)

    def close_gripper(self, wait: bool = True, activate: bool = True) -> None:
        """Close the gripper: ``ActGripper(index, 1)`` then ``MoveGripper(index, closed_percent, ...)``."""
        self._command_gripper(self.cfg.fairino.gripper.closed_percent, 1.0, wait, activate)

    def _command_gripper(self, percent: float, fraction: float, wait: bool, activate: bool) -> None:
        """Move the gripper once, with the same calls as the standalone open/close scripts.

        ``wait`` blocks until the gripper has finished (or ``max_time_ms`` passed), but only while
        arm set-points are not being streamed: a blocked controller link would stall them, so
        during teleoperation the move is always sent without waiting.
        """
        gc = self.cfg.fairino.gripper
        if not self._connected or self._rpc is None:
            raise RobotError("fr10: not connected")
        if not (gc.enabled and self.cfg.gripper.enabled):
            raise RobotError("the gripper is disabled (fairino.gripper.enabled / gripper.enabled)")
        block = 0 if (wait and not self.streaming) else 1
        position = int(round(percent))
        with self._rpc_lock:
            if activate:
                code = self._rpc.ActGripper(gc.index, 1)
                if code != 0:
                    raise RobotError(f"ActGripper({gc.index}, 1) failed: {describe_error(code)}")
            code = self._rpc.MoveGripper(*self._gripper_args(position, block))
        if code != 0:
            raise RobotError(f"MoveGripper({gc.index}, {position}) failed: {describe_error(code)}")
        self.set_gripper(fraction)  # keep the teleoperation command in step with what was just done
        logger.info("[fr10] gripper %s", "closing" if fraction > 0.5 else "opening")

    def _send_gripper(self, position: int, label: str) -> bool:
        """One gripper move, with the calls of gripper.py: ActGripper(index, 1) then MoveGripper(...).

        Sent on the gripper's own connection, never under the arm's lock: a slow gripper command
        must not hold up the arm's set-points. Returns True if the controller accepted it.
        """
        self._gripper_busy_since = time.monotonic()
        try:
            return self._send_gripper_call(position, label)
        finally:
            self._gripper_busy_since = None

    def _send_gripper_call(self, position: int, label: str) -> bool:
        gc = self.cfg.fairino.gripper
        started = time.perf_counter()
        driver = self._gripper_driver
        if driver is not None:
            try:
                driver.move_percent(
                    position,
                    activate=gc.activate_before_move or self._gripper_needs_activation,
                    block=0 if gc.wait_for_move else 1,
                    log=False,
                )
            except Exception as exc:  # noqa: BLE001 - retried by the caller (activating again takes seconds)
                self._gripper_failures += 1
                self._gripper_last = (label, False, str(exc), time.monotonic())
                if self._gripper_failures in (1, 5, 20) or self._gripper_failures % 100 == 0:
                    logger.warning("[fr10] gripper %s failed (%d times): %s", label, self._gripper_failures, exc)
                return False
            took = time.perf_counter() - started
            self._gripper_needs_activation = False
            self._gripper_failures = 0
            self.gripper_call_max = max(self.gripper_call_max, took)
            self._gripper_last = (label, True, "", time.monotonic())
            self._gripper_commands += 1
            self._log_gripper_move(position, label, took)
            return True
        try:
            if gc.activate_before_move:
                code = self._rpc_call("ActGripper", gc.index, 1, timeout=gc.activation_timeout_s)
                if code != 0:
                    raise RobotError(f"ActGripper({gc.index}, 1) answered {describe_error(code)}")
            block = 0 if gc.wait_for_move else 1
            code = self._rpc_call("MoveGripper", *self._gripper_args(position, block))
            if code != 0:
                raise RobotError(f"MoveGripper({gc.index}, {position}) answered {describe_error(code)}")
        except Exception as exc:  # noqa: BLE001 - reported below, retried by the caller
            self._gripper_last = (label, False, str(exc), time.monotonic())
            self._gripper_failures += 1
            if self._gripper_failures in (1, 5, 20) or self._gripper_failures % 100 == 0:
                logger.warning("[fr10] gripper %s failed (%d times): %s", label, self._gripper_failures, exc)
            return False
        took = time.perf_counter() - started
        self.gripper_call_max = max(self.gripper_call_max, took)
        self._gripper_failures = 0
        self._gripper_last = (label, True, "", time.monotonic())
        self._gripper_commands += 1
        self._log_gripper_move(position, label, took)
        return True

    def _log_gripper_move(self, position: int, label: str, took: float) -> None:
        logger.info(
            "[fr10] gripper %s (position %d %%): the controller accepted it in %.0f ms", label, position, 1e3 * took
        )

    def _gripper_loop(self) -> None:
        """Dedicated gripper thread: crashes are reported instead of silently ending it."""
        try:
            self._gripper_loop_body()
        except Exception:  # noqa: BLE001
            logger.exception("[fr10] the gripper thread crashed: the gripper will not move until the program is restarted")

    def _gripper_loop_body(self) -> None:
        gc = self.cfg.fairino.gripper
        logger.info(
            "[fr10] gripper thread started (%s mode, %s, %s)",
            gc.mode,
            "waits for each move" if gc.wait_for_move else "does not wait for moves",
            "own controller connection" if self._gripper_driver is not None else "plain XML-RPC calls",
        )
        state: Optional[str] = None  # open_close: the state the gripper was last commanded to
        last_sent: Optional[int] = None  # proportional: the last position sent
        last_attempt = 0.0
        while not self._gripper_stop.is_set():
            self._gripper_wake.wait(gc.min_interval_s)  # woken at once by a new command
            self._gripper_wake.clear()
            if self._gripper_stop.is_set():
                break
            if not self.streaming:
                state = last_sent = None  # start again from the measured opening next session
                self._gripper_needs_activation = True
                continue
            if time.monotonic() - last_attempt < gc.min_interval_s:
                continue
            command = self.gripper_command  # hand closure 0 (open) .. 1 (closed)
            if self._gripper_resend.is_set():
                self._gripper_resend.clear()
                state = last_sent = None  # forget what was sent: the next block sends the command again
                self._gripper_needs_activation = True  # and activates the gripper first, as gripper.py does
            if gc.mode == "open_close":
                if state is None:
                    # Once at the start of a session: drive the gripper to the state it is believed to be in.
                    # The controller's reported opening can be wrong (a closed gripper that reports "open"
                    # would never be opened otherwise); if the belief is right this moves nothing.
                    wanted = "closed" if command >= 0.5 else "open"
                elif state == "open" and command >= gc.close_above:
                    wanted = "closed"
                elif state == "closed" and command <= gc.open_below:
                    wanted = "open"
                else:
                    continue
                position = int(round(gc.closed_percent if wanted == "closed" else gc.open_percent))
                last_attempt = time.monotonic()
                if self._send_gripper(position, "CLOSE" if wanted == "closed" else "OPEN"):
                    state = wanted  # on failure the same command is tried again next time
            else:
                position = int(round(gc.open_percent + command * (gc.closed_percent - gc.open_percent)))
                at_end = position in (int(round(gc.open_percent)), int(round(gc.closed_percent)))
                if last_sent is not None and (
                    position == last_sent or (abs(position - last_sent) < gc.resolution_percent and not at_end)
                ):
                    continue  # (the first command of a session always goes out: see above)
                last_attempt = time.monotonic()
                if self._send_gripper(position, f"MOVE {position} %"):
                    last_sent = position
