"""Typed configuration for the FR10 teleoperation stack.

The dataclasses below are the single source of truth for every tunable and its default.
``config/default.yaml`` mirrors them for documentation; a YAML file only needs to list the
values it wants to change. Unknown keys are rejected so that typos never go unnoticed.

Angles are written in degrees in the configuration (suffix ``_deg``) and exposed in radians
through ``*_rad`` properties.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

NUM_ARM_JOINTS = 6


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


def project_path(value: Union[str, Path]) -> Path:
    """Expand ``~`` and resolve a relative path against the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rad(values_deg) -> List[float]:
    return [math.radians(float(v)) for v in values_deg]


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------
@dataclass
class RobotConfig:
    """Kinematic model and motion limits of the arm."""

    urdf: str = "assets/fr10/fr10_gripper.urdf"
    arm_joints: List[str] = field(default_factory=lambda: ["j1", "j2", "j3", "j4", "j5", "j6"])
    flange_frame: str = "flange"
    # Tool centre point relative to the flange: x, y, z [m], roll, pitch, yaw [deg].
    tcp_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.130, 0.0, 0.0, 0.0])
    # Ready pose: elbow up, gripper pointing straight down 0.40 m above the table, 0.66 m in
    # front of the base (TCP = [0.664, 0.002, 0.396] m), far from every joint limit.
    home_deg: List[float] = field(default_factory=lambda: [14.0, -90.0, -115.0, -65.0, 90.0, 14.0])
    # Safety margin kept from the URDF joint limits.
    joint_limit_margin_deg: float = 3.0
    # Teleoperation limits (the FR10 itself allows 180 deg/s on every joint).
    max_velocity_deg_s: List[float] = field(default_factory=lambda: [80.0, 80.0, 100.0, 150.0, 150.0, 180.0])
    max_acceleration_deg_s2: List[float] = field(
        default_factory=lambda: [300.0, 300.0, 400.0, 600.0, 600.0, 800.0]
    )

    @property
    def urdf_path(self) -> Path:
        return project_path(self.urdf)

    @property
    def home_rad(self) -> List[float]:
        return _rad(self.home_deg)

    @property
    def max_velocity_rad_s(self) -> List[float]:
        return _rad(self.max_velocity_deg_s)

    @property
    def max_acceleration_rad_s2(self) -> List[float]:
        return _rad(self.max_acceleration_deg_s2)

    @property
    def tcp_offset_xyzrpy(self) -> List[float]:
        return [float(v) for v in self.tcp_offset[:3]] + _rad(self.tcp_offset[3:])

    def validate(self) -> None:
        _require(len(self.arm_joints) == NUM_ARM_JOINTS, "robot.arm_joints must list 6 joints")
        for name in ("home_deg", "max_velocity_deg_s", "max_acceleration_deg_s2", "tcp_offset"):
            _require(len(getattr(self, name)) == NUM_ARM_JOINTS, f"robot.{name} must have 6 values")
        _require(all(v > 0 for v in self.max_velocity_deg_s), "robot.max_velocity_deg_s must be > 0")
        _require(all(v > 0 for v in self.max_acceleration_deg_s2), "robot.max_acceleration_deg_s2 must be > 0")
        _require(0.0 <= self.joint_limit_margin_deg < 30.0, "robot.joint_limit_margin_deg out of range")


@dataclass
class ControlConfig:
    """Loop rates and supervision thresholds."""

    rate_hz: float = 60.0  # XR -> IK loop
    servo_rate_hz: float = 125.0  # joint set-point streaming (8 ms, FAIRINO ServoJ period)
    speed_scale: float = 1.0  # global multiplier on velocity/acceleration limits, (0, 1]
    home_speed_deg_s: float = 25.0  # joint speed used for "go home"
    home_on_start: bool = True  # after 'r', first move the arm to the home (ready) pose, then wait for the clutch
    tracking_gain: float = 15.0  # [1/s] stiffness of the set-point filter near its target
    state_timeout_s: float = 0.5  # robot feedback older than this is a fault
    tracking_error_limit_deg: float = 30.0  # |command - measured| above this is a fault
    # How far the command may run ahead of the measured arm. The arm always lags its command a little
    # (a real arm in servo mode lags by a few tenths of a second), and that must not shape the motion:
    # up to soft_command_lead_deg nothing happens; between soft and max the command is slowed down
    # smoothly (braking stays as strong as ever); at max it waits. A hard, low limit makes the motion
    # stop and go.
    soft_command_lead_deg: float = 10.0
    max_command_lead_deg: float = 22.0
    stall_timeout_s: float = 1.5  # command held back this long without the arm catching up: fault
    auto_start: bool = False  # skip the "press r" gate (testing only)

    def validate(self) -> None:
        _require(5.0 <= self.rate_hz <= 500.0, "control.rate_hz must be within [5, 500]")
        _require(20.0 <= self.servo_rate_hz <= 1000.0, "control.servo_rate_hz must be within [20, 1000]")
        _require(0.0 < self.speed_scale <= 1.0, "control.speed_scale must be within (0, 1]")
        _require(self.home_speed_deg_s > 0.0, "control.home_speed_deg_s must be > 0")
        _require(
            0.0 < self.tracking_gain * (1.0 / self.servo_rate_hz) < 0.5,
            "control.tracking_gain is too high for control.servo_rate_hz",
        )
        _require(self.state_timeout_s > 0.0, "control.state_timeout_s must be > 0")
        _require(1.0 <= self.max_command_lead_deg <= 45.0, "control.max_command_lead_deg must be within [1, 45]")
        _require(
            0.5 <= self.soft_command_lead_deg < self.max_command_lead_deg,
            "need 0.5 <= control.soft_command_lead_deg < control.max_command_lead_deg",
        )
        _require(
            self.max_command_lead_deg < self.tracking_error_limit_deg,
            "control.max_command_lead_deg must be below control.tracking_error_limit_deg "
            "(otherwise a lagging arm is reported as a fault)",
        )
        _require(self.stall_timeout_s > 0.0, "control.stall_timeout_s must be > 0")
        _require(self.tracking_error_limit_deg > 0.0, "control.tracking_error_limit_deg must be > 0")


@dataclass
class IKConfig:
    """Weights of the optimisation-based inverse kinematics."""

    position_weight: float = 100.0  # per m^2
    rotation_weight: float = 5.0  # per rad^2
    posture_weight: float = 0.0001  # weak pull towards the home posture, per rad^2
    smoothness_weight: float = 0.1  # penalty on the change since the previous solution, per rad^2
    max_step_deg: float = 10.0  # largest joint change per solve (prevents configuration flips)
    max_iterations: int = 40
    tolerance: float = 1.0e-6

    def validate(self) -> None:
        for name in ("position_weight", "rotation_weight", "posture_weight", "smoothness_weight"):
            _require(getattr(self, name) >= 0.0, f"ik.{name} must be >= 0")
        _require(self.position_weight > 0.0, "ik.position_weight must be > 0")
        _require(0.1 <= self.max_step_deg <= 90.0, "ik.max_step_deg must be within [0.1, 90]")
        _require(self.max_iterations >= 5, "ik.max_iterations must be >= 5")


@dataclass
class WorkspaceConfig:
    """Region (robot base frame, metres) the tool centre point is allowed to be commanded to."""

    x: List[float] = field(default_factory=lambda: [0.20, 1.15])
    y: List[float] = field(default_factory=lambda: [-0.75, 0.75])
    z: List[float] = field(default_factory=lambda: [0.02, 1.20])
    max_reach: float = 1.25  # distance from the shoulder (FR10 reach is 1.4 m)
    min_radius: float = 0.25  # distance from the base axis

    def validate(self) -> None:
        for axis in ("x", "y", "z"):
            lim = getattr(self, axis)
            _require(len(lim) == 2 and lim[0] < lim[1], f"mapping.workspace.{axis} must be [min, max]")
        _require(0.3 < self.max_reach <= 1.40, "mapping.workspace.max_reach must be within (0.3, 1.40]")
        _require(0.0 <= self.min_radius < self.max_reach, "mapping.workspace.min_radius out of range")


@dataclass
class MappingConfig:
    """How operator hand motion is turned into tool motion (relative, clutched)."""

    position_scale: float = 1.0  # robot metres per hand metre
    rotation_scale: float = 1.0
    track_orientation: bool = True  # False keeps the tool orientation fixed
    # "absolute": the tool orientation follows the hand orientation. Palm down, fingers forward
    #             gives the home orientation (gripper pointing down); turning the hand turns the
    #             gripper the same way. The clutch only closes once the hand roughly matches the
    #             gripper (see engage_tolerance_deg), so the arm never snaps.
    # "relative": only the rotation since the clutch closed is copied (any hand posture starts
    #             with the gripper as it is).
    orientation_mode: str = "absolute"
    engage_tolerance_deg: float = 35.0  # absolute mode: largest hand-vs-gripper mismatch that engages
    # Tuning of the "palm down" reference: extra rotation applied in the hand frame, roll, pitch,
    # yaw in degrees. Leave at zero unless palm-down does not give a downward gripper.
    hand_offset_rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # "head_yaw": directions are relative to where the operator faces when the clutch closes.
    # "world":    directions are fixed in the headset's world frame.
    reference: str = "head_yaw"
    # Rotation (about vertical) from the operator's forward direction to the robot's +x axis.
    # "auto": use the simulator camera heading in simulation, 0 on the real robot.
    yaw_offset_deg: Union[float, str] = "auto"
    # Low-pass on the target pose. The time constant is `smoothing_time_s` while the hand is
    # almost still (this removes the tracking jitter of a bare hand) and shrinks as the hand
    # moves faster, so deliberate motion is not delayed: tau = smoothing_time_s / (1 + speed / ref).
    smoothing_time_s: float = 0.12
    smoothing_fast_speed_m_s: float = 0.25  # hand speed at which the smoothing time has halved
    smoothing_fast_turn_deg_s: float = 90.0  # same for rotation
    max_linear_speed: float = 0.6  # m/s, limit on target motion
    max_angular_speed_deg: float = 150.0
    jump_position_m: float = 0.15  # larger hand jumps between two cycles drop the clutch
    jump_rotation_deg: float = 50.0
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)

    def validate(self) -> None:
        _require(0.05 <= self.position_scale <= 5.0, "mapping.position_scale must be within [0.05, 5]")
        _require(0.0 <= self.rotation_scale <= 2.0, "mapping.rotation_scale must be within [0, 2]")
        _require(self.reference in ("head_yaw", "world"), "mapping.reference must be 'head_yaw' or 'world'")
        _require(
            self.orientation_mode in ("absolute", "relative"),
            "mapping.orientation_mode must be 'absolute' or 'relative'",
        )
        _require(0.0 < self.engage_tolerance_deg <= 180.0, "mapping.engage_tolerance_deg must be within (0, 180]")
        _require(len(self.hand_offset_rpy_deg) == 3, "mapping.hand_offset_rpy_deg must have 3 values")
        if isinstance(self.yaw_offset_deg, str):
            _require(self.yaw_offset_deg == "auto", "mapping.yaw_offset_deg must be a number or 'auto'")
        _require(self.smoothing_time_s >= 0.0, "mapping.smoothing_time_s must be >= 0")
        _require(self.smoothing_fast_speed_m_s > 0.0, "mapping.smoothing_fast_speed_m_s must be > 0")
        _require(self.smoothing_fast_turn_deg_s > 0.0, "mapping.smoothing_fast_turn_deg_s must be > 0")
        _require(self.max_linear_speed > 0.0, "mapping.max_linear_speed must be > 0")
        _require(self.max_angular_speed_deg > 0.0, "mapping.max_angular_speed_deg must be > 0")
        _require(self.jump_position_m > 0.0 and self.jump_rotation_deg > 0.0, "mapping.jump_* must be > 0")
        self.workspace.validate()


@dataclass
class XRConfig:
    """Headset input and display."""

    input_mode: str = "controller"  # "controller" (recommended) or "hand"
    hand: str = "right"  # which controller / hand drives the arm
    # "immersive": camera image fills the view, "ego": small camera window + real world,
    # "pass-through": real world only, "auto": immersive in simulation, pass-through otherwise.
    display_mode: str = "auto"
    display_fps: float = 30.0
    clutch_on: float = 0.6  # grip (squeeze) value that closes the clutch
    clutch_off: float = 0.3  # grip value that opens it again (hysteresis)
    stale_timeout_s: float = 0.4  # no new pose for this long -> tracking lost
    # Hand mode clutch: a closed fist of the non-driving hand. The fist is measured as the mean ratio
    # (fingertip distance from the wrist) / (knuckle distance from the wrist) over the four fingers:
    # about 2.0 for an open hand, about 1.0 for a fist.
    fist_closed_ratio: float = 1.25  # ratio at or below which the hand counts as a fist
    fist_open_ratio: float = 1.50  # ratio at or above which it counts as open again (hysteresis)
    clutch_engage_delay_s: float = 0.10  # the fist must be held this long before the clutch closes
    clutch_release_delay_s: float = 0.12  # opening (or losing) the hand must last this long to release
    # Hand mode gripper.
    #   "driving_hand_fist":  how far the driving hand is closed sets the gripper opening
    #                         continuously: open hand = open, half closed = half closed, fist = closed.
    #                         A thumb-index pinch of the same hand closes it too (gripper_pinch_*).
    # Pinch options (a pinch is thumb and index fingertip touching with the other three fingers
    # straight; a fist is the clutch, so the two gestures cannot be confused):
    #   "both_hands_pinch":   a pinch of either hand toggles the gripper open/closed.
    #   "other_hand_pinch":   only a pinch of the non-driving (clutch) hand toggles it.
    #   "driving_hand_pinch": thumb-index distance of the driving hand sets the opening continuously.
    gripper_source: str = "driving_hand_fist"
    # Hand mode: the hand joint positions are also published on this local port (0 = off) so that
    # tools/pinch_gripper.py --from-teleop can drive the gripper in its own process.
    hand_bus_port: int = 5560
    gripper_fist_open_ratio: float = 1.8  # hand closure ratio (see fist_*_ratio) of a fully open gripper
    gripper_fist_closed_ratio: float = 1.15  # ... and of a fully closed gripper (a fist)
    gripper_pinch_open_m: float = 0.08  # thumb-index distance at which a pinch starts to close the gripper
    gripper_pinch_closed_m: float = 0.02  # ... and at which it is fully closed (half way: 50 mm)
    gripper_pinch_on_m: float = 0.03  # thumb-index distance below which a pinch is recognised
    gripper_pinch_off_m: float = 0.05  # distance above which it ends (hysteresis)
    gripper_pinch_hold_s: float = 0.08  # a pinch must last this long (ignores tracking glitches)
    pinch_open_m: float = 0.09  # driving-hand mode: thumb-index distance for a fully open gripper
    pinch_closed_m: float = 0.02  # hand mode: thumb-index distance for a fully closed gripper
    cert_file: Optional[str] = None  # HTTPS certificate (None -> televuer defaults)
    key_file: Optional[str] = None

    @property
    def cert_path(self) -> Optional[Path]:
        return project_path(self.cert_file) if self.cert_file else None

    @property
    def key_path(self) -> Optional[Path]:
        return project_path(self.key_file) if self.key_file else None

    def validate(self) -> None:
        _require(self.input_mode in ("controller", "hand"), "xr.input_mode must be 'controller' or 'hand'")
        _require(self.hand in ("left", "right"), "xr.hand must be 'left' or 'right'")
        _require(
            self.display_mode in ("auto", "immersive", "ego", "pass-through"),
            "xr.display_mode must be auto, immersive, ego or pass-through",
        )
        _require(0.0 < self.clutch_off < self.clutch_on < 1.0, "need 0 < xr.clutch_off < xr.clutch_on < 1")
        _require(self.pinch_closed_m < self.pinch_open_m, "need xr.pinch_closed_m < xr.pinch_open_m")
        _require(
            0.0 < self.fist_closed_ratio < self.fist_open_ratio,
            "need 0 < xr.fist_closed_ratio < xr.fist_open_ratio",
        )
        _require(
            self.clutch_engage_delay_s >= 0.0 and self.clutch_release_delay_s >= 0.0,
            "xr.clutch_engage_delay_s and xr.clutch_release_delay_s must be >= 0",
        )
        _require(
            self.gripper_source in ("driving_hand_fist", "both_hands_pinch", "other_hand_pinch", "driving_hand_pinch"),
            "xr.gripper_source must be 'driving_hand_fist', 'both_hands_pinch', 'other_hand_pinch' "
            "or 'driving_hand_pinch'",
        )
        _require(
            0.0 < self.gripper_pinch_closed_m < self.gripper_pinch_open_m,
            "need 0 < xr.gripper_pinch_closed_m < xr.gripper_pinch_open_m",
        )
        _require(
            0.0 < self.gripper_fist_closed_ratio < self.gripper_fist_open_ratio,
            "need 0 < xr.gripper_fist_closed_ratio < xr.gripper_fist_open_ratio",
        )
        _require(
            0.0 < self.gripper_pinch_on_m < self.gripper_pinch_off_m,
            "need 0 < xr.gripper_pinch_on_m < xr.gripper_pinch_off_m",
        )
        _require(self.gripper_pinch_hold_s >= 0.0, "xr.gripper_pinch_hold_s must be >= 0")
        _require(0 <= self.hand_bus_port <= 65535, "xr.hand_bus_port must be within [0, 65535]")
        _require(self.stale_timeout_s > 0.0, "xr.stale_timeout_s must be > 0")


@dataclass
class GripperConfig:
    """Gripper command shaping (backend independent). 0 = open, 1 = closed."""

    enabled: bool = True
    mode: str = "analog"  # "analog": follows the trigger, "toggle": each trigger press flips open/closed
    toggle_threshold: float = 0.7
    max_rate: float = 4.0  # full strokes per second

    def validate(self) -> None:
        _require(self.mode in ("analog", "toggle"), "gripper.mode must be 'analog' or 'toggle'")
        _require(0.0 < self.toggle_threshold < 1.0, "gripper.toggle_threshold must be within (0, 1)")
        _require(self.max_rate > 0.0, "gripper.max_rate must be > 0")


@dataclass
class ImageConfig:
    """Camera stream shown in the headset and stored in recordings (raw JPEG frames over ZeroMQ)."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 55555
    shape: List[int] = field(default_factory=lambda: [480, 640])  # [height, width] of the full frame
    binocular: bool = False  # True: frame holds left|right eye images side by side

    def validate(self) -> None:
        _require(len(self.shape) == 2 and all(int(v) > 0 for v in self.shape), "image.shape must be [h, w]")
        _require(not self.binocular or self.shape[1] % 2 == 0, "binocular image width must be even")
        _require(1 <= self.port <= 65535, "image.port must be within [1, 65535]")


@dataclass
class OrbbecConfig:
    """Gemini 2 color capture for camera_fr10.py (SDK imported only by the camera server)."""

    serial: Optional[str] = None  # None selects the only connected camera
    width: int = 640
    height: int = 480
    fps: int = 30
    jpeg_quality: int = 80
    frame_timeout_ms: int = 1000

    def validate(self) -> None:
        _require(self.serial is None or bool(self.serial.strip()), "orbbec.serial must be nonempty or null")
        _require(self.width > 0 and self.height > 0, "orbbec.width and height must be > 0")
        _require(1 <= self.fps <= 120, "orbbec.fps must be within [1, 120]")
        _require(1 <= self.jpeg_quality <= 100, "orbbec.jpeg_quality must be within [1, 100]")
        _require(self.frame_timeout_ms > 0, "orbbec.frame_timeout_ms must be > 0")


@dataclass
class SimCameraConfig:
    eye: List[float] = field(default_factory=lambda: [-0.85, -0.60, 1.25])  # base frame [m]
    target: List[float] = field(default_factory=lambda: [0.60, 0.0, 0.10])
    focal_length_mm: float = 16.0  # with a 20.955 mm aperture: 66 deg horizontal field of view
    rate_hz: float = 30.0
    jpeg_quality: int = 80


@dataclass
class SimConfig:
    """Isaac Lab simulator (sim_fr10.py) and the link to it."""

    host: str = "127.0.0.1"  # where the simulator runs, seen from the teleop program
    state_port: int = 5556
    command_port: int = 5557
    physics_hz: float = 200.0
    render_every: int = 4  # render one frame every N physics steps
    command_timeout_s: float = 0.5  # hold position if the teleop program goes silent
    table_height: float = 0.75
    cube_size: float = 0.05
    camera: SimCameraConfig = field(default_factory=SimCameraConfig)

    def validate(self) -> None:
        _require(60.0 <= self.physics_hz <= 1000.0, "sim.physics_hz must be within [60, 1000]")
        _require(self.render_every >= 1, "sim.render_every must be >= 1")
        _require(self.state_port != self.command_port, "sim.state_port and sim.command_port must differ")
        _require(len(self.camera.eye) == 3 and len(self.camera.target) == 3, "sim.camera eye/target need 3 values")
        _require(1 <= self.camera.jpeg_quality <= 100, "sim.camera.jpeg_quality must be within [1, 100]")


@dataclass
class FairinoGripperConfig:
    """Gripper connected to the FAIRINO controller (ActGripper / MoveGripper)."""

    enabled: bool = True
    index: int = 2  # gripper number configured on the teach pendant
    open_percent: float = 0.0  # MoveGripper position for "open"
    closed_percent: float = 100.0  # MoveGripper position for "closed"
    speed_percent: int = 100
    force_percent: int = 67
    max_time_ms: int = 2000
    # Every gripper command shares the controller with the arm's joint commands; many commands stutter the arm.
    min_interval_s: float = 0.4  # shortest time between two gripper commands
    resolution_percent: float = 10.0  # proportional mode: smallest position change worth sending
    # "proportional": the gripper goes to the position that matches the hand closure, with the calls of
    #   the standalone gripper.py (MoveGripper(index, 0..100, speed, force, ...)): a half pinch stops
    #   the gripper halfway, a full pinch or fist closes it fully, opening the hand opens it.
    # "open_close": only open (0) or closed (100): closes above close_above, opens below open_below.
    mode: str = "proportional"
    close_above: float = 0.6  # open_close: hand closure at or above which the gripper closes
    open_below: float = 0.4  # open_close: hand closure at or below which the gripper opens
    # ActGripper(index, 1) is always called before the first move of a session and after a failed
    # move; True also calls it before every single move, as gripper.py does.
    activate_before_move: bool = False
    # block=0 as in gripper.py: the gripper thread waits for each move to finish (up to max_time_ms). It has
    # its own connection and thread, so the arm is not held up. False sends the move and does not wait.
    wait_for_move: bool = False
    activation_timeout_s: float = 20.0  # ActGripper can take many seconds on a gripper that is not active yet
    # True opens a second vendor-SDK connection just for the gripper. The controller may allow only one such
    # client, and the SDK then waits for ever without a message. False (default): plain XML-RPC calls with
    # a timeout on the gripper's own socket, the same calls with the same argument types as the SDK sends.
    separate_sdk_connection: bool = False
    # True (default): the gripper runs in a process of its own with a shared target value, like the Inspire
    # hand controller of the original project: it activates the gripper once when the program connects,
    # then follows the target at a fixed low rate, and cannot be held up by (or hold up) the arm's loop.
    use_process: bool = True
    # Measured on the FR10 controller (V3.9.4): MoveGripper runs in the same motion queue as the ServoJ
    # set-points. While a gripper move runs (~1.2 s for a full stroke) the arm stands still and the set-points
    # streamed meanwhile pile up behind it, so the arm then lags ~1.3 s for the rest of the session, and every
    # later MoveGripper waits for a queue that never empties (they all run when teleoperation ends).
    # True (only with use_process): for each gripper move during teleoperation the arm brakes smoothly, sends
    # nothing until the controller reports the gripper move done (the queue stays empty: no added lag), then
    # follows the hand again. Proportional mode sends a position only once the fingers have settled, or at
    # fully open / closed, so that is one short pause per grasp and one per release.
    pause_arm_for_moves: bool = True
    pause_cooldown_s: float = 1.0  # shortest time between the end of one paused move and the next one
    still_band_percent: float = 4.0  # hand-closure jitter (percent) that still counts as "fingers settled"
    pause_timeout_s: float = 8.0  # safety: longest arm pause for one move; the arm then follows regardless

    def validate(self) -> None:
        _require(0 <= self.index <= 15, "fairino.gripper.index must be within [0, 15]")
        _require(self.pause_cooldown_s >= 0.0, "fairino.gripper.pause_cooldown_s must be >= 0")
        _require(0.0 <= self.still_band_percent <= 20.0, "fairino.gripper.still_band_percent must be within [0, 20]")
        for name in ("open_percent", "closed_percent", "speed_percent", "force_percent"):
            _require(0 <= getattr(self, name) <= 100, f"fairino.gripper.{name} must be within [0, 100]")
        _require(0 < self.max_time_ms <= 30000, "fairino.gripper.max_time_ms must be within (0, 30000]")
        _require(self.min_interval_s >= 0.02, "fairino.gripper.min_interval_s must be >= 0.02")
        _require(self.resolution_percent >= 1.0, "fairino.gripper.resolution_percent must be >= 1")
        _require(self.activation_timeout_s > 0.0, "fairino.gripper.activation_timeout_s must be > 0")
        _require(self.mode in ("open_close", "proportional"), "fairino.gripper.mode must be open_close or proportional")
        _require(
            0.0 < self.open_below < self.close_above < 1.0,
            "need 0 < fairino.gripper.open_below < fairino.gripper.close_above < 1",
        )


@dataclass
class FairinoConfig:
    """Real FR10 through the FAIRINO Python SDK (Robot.RPC)."""

    ip: str = "192.168.58.2"
    rpc_port: int = 20003  # the controller's XML-RPC command port
    # Directory that contains the SDK's Robot.py / Robot.*.so (the vendor "fairino" folder).
    sdk_path: str = "~/Projects/FoundationPose/cube_pick/robot/fairino_driver"
    # How ServoJ set-points reach the controller: "udp" (the SDK's UDP pass-through on port 20007: fire and
    # forget like the DDS commands of the original project, and it does not share the controller's XML-RPC
    # channel with gripper and set-up commands, so those cannot stall the arm; needs controller firmware
    # that supports it) or "xmlrpc" (every set-point waits for the controller's reply; all firmware).
    servo_transport: str = "udp"
    use_servo_move_start_end: bool = True  # wrap streaming in ServoMoveStart / ServoMoveEnd
    switch_to_auto_mode: bool = True  # call Mode(0) if the robot is in manual mode
    enable_on_connect: bool = True  # call RobotEnable(1)
    reset_errors_on_connect: bool = False  # call ResetAllError() if a fault is active at start-up
    rpc_timeout_s: float = 5.0
    # Latency guard. The controller runs ServoJ set-points (and gripper moves) from one motion queue; the
    # state packet reports its length (mc_queue_len, normally 0-1 while streaming). Anything that stalls
    # the queue makes set-points pile up, and since they arrive as fast as they run, that backlog (= arm
    # lag) would never shrink. While more than this many are queued no new set-points are sent (0: off).
    max_queue_points: int = 10
    gripper: FairinoGripperConfig = field(default_factory=FairinoGripperConfig)

    @property
    def sdk_dir(self) -> Path:
        return project_path(self.sdk_path)

    def validate(self) -> None:
        _require(self.servo_transport in ("xmlrpc", "udp"), "fairino.servo_transport must be 'xmlrpc' or 'udp'")
        _require(self.rpc_timeout_s > 0.0, "fairino.rpc_timeout_s must be > 0")
        _require(0 < self.rpc_port <= 65535, "fairino.rpc_port must be within [1, 65535]")
        _require(0 <= self.max_queue_points <= 1000, "fairino.max_queue_points must be within [0, 1000]")
        self.gripper.validate()
        _require(
            self.gripper.pause_timeout_s > self.rpc_timeout_s + 2.5,
            "fairino.gripper.pause_timeout_s must exceed fairino.rpc_timeout_s + 2.5 s (braking plus the gripper "
            "call's own timeout), or the arm could move on while a gripper move is still waiting",
        )


@dataclass
class RecordingConfig:
    directory: str = "data"  # relative to the project root unless absolute
    task: str = "fr10_teleop"
    rate_hz: float = 30.0
    save_images: bool = True
    jpeg_quality: int = 90

    @property
    def directory_path(self) -> Path:
        return project_path(self.directory)

    def validate(self) -> None:
        _require(self.rate_hz > 0.0, "recording.rate_hz must be > 0")
        _require(1 <= self.jpeg_quality <= 100, "recording.jpeg_quality must be within [1, 100]")


@dataclass
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    xr: XRConfig = field(default_factory=XRConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    orbbec: OrbbecConfig = field(default_factory=OrbbecConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    fairino: FairinoConfig = field(default_factory=FairinoConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def validate(self) -> "Config":
        for section in dataclasses.fields(self):
            getattr(self, section.name).validate()
        _require(
            self.control.servo_rate_hz >= self.control.rate_hz,
            "control.servo_rate_hz must be >= control.rate_hz",
        )
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Convert a YAML value to the annotated field type, rejecting obvious mismatches."""
    origin = typing.get_origin(annotation)
    if origin is Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if value is None:
            _require(len(args) < len(typing.get_args(annotation)), f"'{path}' must not be null")
            return None
        errors = []
        # Prefer an exact type match so that e.g. the string "auto" is not parsed as a number.
        for arg in sorted(args, key=lambda a: not isinstance(value, a) if isinstance(a, type) else True):
            try:
                return _coerce(value, arg, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(errors[0])
    if origin in (list, List):
        _require(isinstance(value, (list, tuple)), f"'{path}' must be a list")
        (item_type,) = typing.get_args(annotation) or (Any,)
        return [_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]
    if dataclasses.is_dataclass(annotation):
        return _build(annotation, value, path)
    if annotation is bool:
        _require(isinstance(value, bool), f"'{path}' must be true or false")
        return value
    if annotation is int:
        _require(isinstance(value, int) and not isinstance(value, bool), f"'{path}' must be an integer")
        return value
    if annotation is float:
        _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"'{path}' must be a number")
        return float(value)
    if annotation is str:
        _require(isinstance(value, str), f"'{path}' must be a string")
        return value
    return value


def _build(cls, data: Any, path: str):
    if data is None:
        data = {}
    _require(isinstance(data, dict), f"'{path or 'config'}' must be a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    _require(not unknown, f"unknown key(s) {unknown} in '{path or 'config'}' (valid: {sorted(known)})")
    kwargs = {name: _coerce(data[name], hints[name], f"{path}.{name}" if path else name) for name in data}
    return cls(**kwargs)


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def set_by_path(tree: dict, dotted_key: str, value: Any) -> None:
    """Set ``tree['a']['b']['c'] = value`` for ``dotted_key = 'a.b.c'``."""
    keys = dotted_key.split(".")
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        _require(isinstance(node, dict), f"cannot set '{dotted_key}': '{key}' is not a section")
    node[keys[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``section.key=value`` command line override (value is YAML)."""
    _require("=" in text, f"override '{text}' must look like section.key=value")
    key, raw = text.split("=", 1)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse value of override '{text}': {exc}") from exc
    return key.strip(), value


def load_config(path: Union[str, Path, None] = None, overrides: Optional[dict] = None) -> Config:
    """Load the configuration.

    Args:
        path: YAML file. ``None`` loads ``config/default.yaml`` if present, else built-in defaults.
        overrides: nested dictionary (or flat ``{"a.b": value}``) applied on top of the file.
    """
    data: dict = {}
    if path is None and DEFAULT_CONFIG_PATH.exists():
        path = DEFAULT_CONFIG_PATH
    if path is not None:
        path = Path(path).expanduser()
        _require(path.is_file(), f"configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as stream:
            try:
                loaded = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ConfigError(f"cannot parse {path}: {exc}") from exc
        _require(loaded is None or isinstance(loaded, dict), f"{path} must contain a mapping at top level")
        data = loaded or {}
    if overrides:
        nested: dict = {}
        for key, value in overrides.items():
            if isinstance(value, dict) and "." not in key:
                _deep_update(nested.setdefault(key, {}), value)
            else:
                set_by_path(nested, key, value)
        _deep_update(data, nested)
    return _build(Config, data, "").validate()
