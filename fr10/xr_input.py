"""Operator input: headset controller / hand tracking, or a scripted stand-in.

Every source returns an :class:`OperatorInput`. Poses are expressed in the headset's world
frame using the robot axis convention (x forward, y left, z up); the raw WebXR data (x right,
y up, z backward) is converted here and nowhere else.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Array, Value
from typing import Callable, Optional

import cv2
import numpy as np

from .config import Config
from .mapping import PALM_DOWN
from .math_utils import is_valid_pose, make_pose, rot_y, rot_z

logger = logging.getLogger(__name__)

# Change of basis from WebXR/OpenXR axes (x right, y up, z back) to robot axes (x fwd, y left, z up).
T_ROBOT_OPENXR = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# OpenXR wrist axes (x toward the fingers, y out of the back of the hand, z backwards along the arm)
# to the arm convention used by the mapping (see PALM_DOWN in mapping.py). Same constants as televuer.
T_ARM_FROM_WRIST = {
    "right": np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]]),
    "left": np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]]),
}


def openxr_to_robot(pose_xr: np.ndarray) -> np.ndarray:
    """Re-express an OpenXR pose in robot axes (similarity transform, as in televuer)."""
    return T_ROBOT_OPENXR @ pose_xr @ T_ROBOT_OPENXR.T


@dataclass
class OperatorInput:
    connected: bool = False  # the device has delivered motion data at least once
    valid: bool = False  # hand_pose is valid and fresh
    hand_pose: Optional[np.ndarray] = None  # 4x4, controller or wrist pose
    head_pose: Optional[np.ndarray] = None  # 4x4, may be None
    clutch: float = 0.0  # 0..1, the arm follows while this is high
    grip: float = 0.0  # 0..1 requested gripper closing
    button_a: bool = False  # toggle recording
    button_b: bool = False  # go home
    detail: str = ""  # device specific note shown in status messages


class InputSource(ABC):
    """Where operator commands come from."""

    @abstractmethod
    def read(self) -> OperatorInput: ...

    def diagnostics(self) -> str:
        """Short human-readable note about the state of the input device (for status messages)."""
        return ""

    def show_image(self, bgr: np.ndarray) -> None:
        """Display a camera frame to the operator (no-op if the source has no display)."""

    @property
    def wants_images(self) -> bool:
        return False

    def close(self) -> None:
        pass


# Joint indices of the 25-joint WebXR hand skeleton.
WRIST_JOINT = 0
KNUCKLE_JOINTS = (6, 11, 16, 21)  # proximal phalanx of index, middle, ring and little finger
FINGERTIP_JOINTS = (9, 14, 19, 24)
THUMB_TIP_JOINT = 4
INDEX_TIP_JOINT = 9


def finger_ratios(positions) -> Optional[np.ndarray]:
    """Per finger (index, middle, ring, little): (tip-to-wrist) / (knuckle-to-wrist) distance.

    Scale and orientation independent: about 2.0 for a straight finger, about 1.0 for a curled one.
    Returns ``None`` if the hand is not tracked (non-finite, or joints collapsed on the wrist).
    """
    points = np.asarray(positions, dtype=float)
    if points.shape != (25, 3) or not np.all(np.isfinite(points)):
        return None
    wrist = points[WRIST_JOINT]
    knuckle = np.linalg.norm(points[list(KNUCKLE_JOINTS)] - wrist, axis=1)
    tip = np.linalg.norm(points[list(FINGERTIP_JOINTS)] - wrist, axis=1)
    if np.any(knuckle < 0.02):  # an untracked hand reports all joints at the origin
        return None
    return tip / knuckle


def fist_ratio(positions) -> Optional[float]:
    """How closed a hand is: the mean of :func:`finger_ratios` (about 2.0 open, about 1.0 a fist)."""
    ratios = finger_ratios(positions)
    return None if ratios is None else float(np.mean(ratios))


EVENT_OUTCOMES = ("accepted", "malformed_data", "bad_pose", "bad_state", "wrong_mode")

EVENT_REASON_TEXT = {
    "malformed_data": "event data missing or not finite",
    "bad_pose": "the selected controller/hand pose is not a valid transform (not tracked?)",
    "bad_state": "button values missing or not finite",
    "wrong_mode": "events of the other input type (hand vs controller)",
}


class _MotionReceiptMixin:
    """Timestamp valid selected-side XR events in TeleVuer's server process.

    TeleVuer registers ``self.on_*_move`` and then forks its server. Allocate
    the shared value before its constructor starts that process so the parent
    control loop observes receipts even when the pose is perfectly still.
    """

    def __init__(self, *, fr10_side: str, use_hand_tracking: bool, **kwargs):
        self._fr10_side = fr10_side
        self._fr10_hand_mode = use_hand_tracking
        self._fr10_motion_stamp = Value("d", 0.0, lock=True)
        self._fr10_event_counts = Array("i", len(EVENT_OUTCOMES), lock=True)
        super().__init__(use_hand_tracking=use_hand_tracking, **kwargs)

    @property
    def fr10_event_counts(self) -> dict:
        """How many headset events were accepted / rejected, and why."""
        with self._fr10_event_counts.get_lock():
            return dict(zip(EVENT_OUTCOMES, self._fr10_event_counts[:]))

    def _count_event(self, outcome: str) -> None:
        with self._fr10_event_counts.get_lock():
            self._fr10_event_counts[EVENT_OUTCOMES.index(outcome)] += 1

    @property
    def fr10_motion_time(self) -> float:
        with self._fr10_motion_stamp.get_lock():
            return self._fr10_motion_stamp.value

    def _classify_event(self, event, hand_mode: bool) -> str:
        """Name of the outcome for one headset event ("accepted" or the reason it is rejected)."""
        if hand_mode != self._fr10_hand_mode:
            return "wrong_mode"
        try:
            value = event.value
            length = 25 * 16 if hand_mode else 16
            for side in ("left", "right"):
                # The upstream callbacks require both flat arrays and states.
                # An untracked opposite side may contain zeros; only the
                # selected side needs a valid rotation/translation matrix.
                data = np.asarray(value[side], dtype=float)
                if data.shape != (length,) or not np.all(np.isfinite(data)):
                    return "malformed_data"
                if side == self._fr10_side:
                    pose = data[:16].reshape(4, 4, order="F")
                    if not is_valid_pose(pose) or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
                        return "bad_pose"
                state = value[f"{side}State"]
                if not isinstance(state, dict):
                    return "bad_state"
                scalars = ("pinchValue", "squeezeValue") if hand_mode else ("triggerValue", "squeezeValue")
                if not all(math.isfinite(float(state.get(key, 0.0))) for key in scalars):
                    return "bad_state"
                if not hand_mode:
                    stick = np.asarray(state.get("thumbstickValue", [0.0, 0.0]), dtype=float)
                    if stick.shape != (2,) or not np.all(np.isfinite(stick)):
                        return "bad_state"
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return "malformed_data"
        return "accepted"

    def _valid_motion_event(self, event, hand_mode: bool) -> bool:
        outcome = self._classify_event(event, hand_mode)
        self._count_event(outcome)
        return outcome == "accepted"

    def _stamp_motion(self) -> None:
        with self._fr10_motion_stamp.get_lock():
            self._fr10_motion_stamp.value = time.monotonic()

    async def on_controller_move(self, event, session, fps=60):
        if self._valid_motion_event(event, hand_mode=False):
            await super().on_controller_move(event, session, fps=fps)
            self._stamp_motion()

    async def on_hand_move(self, event, session, fps=60):
        if self._valid_motion_event(event, hand_mode=True):
            await super().on_hand_move(event, session, fps=fps)
            self._stamp_motion()


class TeleVuerInput(InputSource):
    """Meta Quest (or any WebXR headset) through the ``televuer`` module of xr_teleoperate."""

    def __init__(self, cfg: Config, display_mode: str):
        from televuer import TeleVuer  # imported lazily: starts a web server process

        xr = cfg.xr
        if display_mode not in ("immersive", "ego", "pass-through"):
            raise ValueError(f"unknown display mode '{display_mode}'")
        self._cfg = xr
        self._hand_mode = xr.input_mode == "hand"
        self._side = xr.hand
        self._other = "left" if xr.hand == "right" else "right"
        self._display_mode = display_mode
        self._shape = (int(cfg.image.shape[0]), int(cfg.image.shape[1]))
        # Keep the optional TeleVuer import lazy while overriding the callbacks
        # before its constructor registers them with the server.
        receipt_televuer = type("_ReceiptTeleVuer", (_MotionReceiptMixin, TeleVuer), {})
        self._pinch_state = {
            "left": {"on": False, "since": None},
            "right": {"on": False, "since": None},
        }
        self._grip_value = 0.0  # low-passed closure of the driving hand
        self._grip_time = None
        self._clutch_on = False
        self._clutch_since = None  # when the fist (or the open hand) was first seen in the current state
        self._tv = receipt_televuer(
            fr10_side=self._side,
            use_hand_tracking=self._hand_mode,
            binocular=cfg.image.binocular,
            img_shape=self._shape,
            display_fps=xr.display_fps,
            display_mode=display_mode,
            zmq=display_mode != "pass-through",  # frames are pushed with show_image()
            webrtc=False,
            cert_file=str(xr.cert_path) if xr.cert_path else None,
            key_file=str(xr.key_path) if xr.key_path else None,
        )

    @property
    def wants_images(self) -> bool:
        return self._display_mode != "pass-through"

    def show_image(self, bgr: np.ndarray) -> None:
        if not self.wants_images or bgr is None:
            return
        if bgr.shape[:2] != self._shape:
            bgr = cv2.resize(bgr, (self._shape[1], self._shape[0]), interpolation=cv2.INTER_AREA)
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        self._tv.render_to_xr(np.ascontiguousarray(bgr[:, :, :3]))

    def hand_snapshot(self, side: str):
        """Raw hand data for tools that do not need the whole teleoperation pipeline.

        Returns ``(joint positions (25, 3) or None, seconds since the last valid event or None)``.
        Positions are ``None`` while the headset has not delivered usable data for that hand.
        """
        tv = self._tv
        if not tv.motion_data_ready:
            return None, None
        stamp = tv.fr10_motion_time
        age = time.monotonic() - stamp if stamp > 0.0 else None
        positions = np.asarray(getattr(tv, f"{side}_hand_positions"), dtype=float)
        pose_ok = is_valid_pose(np.asarray(getattr(tv, f"{side}_arm_pose"), dtype=float))
        return (positions if pose_ok else None), age

    def _fist_grip(self, tv, now: float, out: OperatorInput) -> float:
        """Gripper closing 0..1 from the driving hand: open 0, half closed 0.5, fist or pinch 1.

        The closure is the larger of two measures: how far the fingers are curled (a fist closes
        the gripper fully, a half-closed hand halfway) and how close thumb and index fingertip are
        (a pinch closes it, and a loose pinch closes it partly). Low-passed (time constant 0.1 s)
        so tracking noise does not move the gripper. If the hand is not tracked the last value is
        kept: the gripper never opens by itself with a part in it.
        """
        cfg = self._cfg
        side = self._side
        positions = np.asarray(getattr(tv, f"{side}_hand_positions"), dtype=float)
        ratio = fist_ratio(positions)
        pose_ok = is_valid_pose(np.asarray(getattr(tv, f"{side}_arm_pose"), dtype=float))
        if ratio is None and pose_ok:
            # The finger joints are unusable but the hand is tracked: fall back to the gestures
            # the headset itself recognises (pinch / squeeze). Unset values are 0 or False.
            squeeze = float(getattr(tv, f"{side}_hand_squeezeValue", 0.0) or 0.0)
            pinched = bool(getattr(tv, f"{side}_hand_pinch", False))
            pinch_value = float(getattr(tv, f"{side}_hand_pinchValue", 0.0) or 0.0)
            from_distance = 0.0
            if 0.003 < pinch_value < cfg.gripper_pinch_open_m:  # 0 means "no data"
                from_distance = (cfg.gripper_pinch_open_m - pinch_value) / (cfg.gripper_pinch_open_m - cfg.gripper_pinch_closed_m)
            raw = float(np.clip(max(squeeze, 1.0 if pinched else 0.0, from_distance), 0.0, 1.0))
            out.detail += f" | {side} finger joints unusable ({self._joint_problem(positions)}); using headset gestures"
            return self._smooth_grip(raw, now)
        if ratio is None or not pose_ok:
            out.detail += (
                f" | {side} hand closure unknown (holding {100 * self._grip_value:.0f} %): "
                + ("hand pose not valid" if not pose_ok else self._joint_problem(positions))
            )
            return self._grip_value
        fist = float(np.clip((cfg.gripper_fist_open_ratio - ratio) / (cfg.gripper_fist_open_ratio - cfg.gripper_fist_closed_ratio), 0.0, 1.0))
        pinch_distance = float(np.linalg.norm(positions[THUMB_TIP_JOINT] - positions[INDEX_TIP_JOINT]))
        pinch = float(np.clip((cfg.gripper_pinch_open_m - pinch_distance) / (cfg.gripper_pinch_open_m - cfg.gripper_pinch_closed_m), 0.0, 1.0))
        out.detail += f" | {side} pinch {1e3 * pinch_distance:.0f} mm, fingers {ratio:.2f}"
        value = self._smooth_grip(max(fist, pinch), now)
        out.detail += f" | {side} hand closed {100 * value:.0f} %"
        return value

    def _smooth_grip(self, raw: float, now: float) -> float:
        if self._grip_time is None:
            self._grip_value = raw
        else:
            dt = max(0.0, now - self._grip_time)
            self._grip_value += (1.0 - math.exp(-dt / 0.1)) * (raw - self._grip_value)
        self._grip_time = now
        return self._grip_value

    @staticmethod
    def _joint_problem(positions: np.ndarray) -> str:
        """Why the finger joints cannot be used, in a few words (for the status line)."""
        if positions.shape != (25, 3):
            return f"joint array has shape {positions.shape}"
        if not np.all(np.isfinite(positions)):
            return "joint positions are not finite"
        knuckle = np.linalg.norm(positions[list(KNUCKLE_JOINTS)] - positions[WRIST_JOINT], axis=1)
        return f"knuckles only {1e3 * float(np.min(knuckle)):.0f} mm from the wrist"

    def _pinch_gesture(self, tv, now: float, out: OperatorInput, side: str) -> bool:
        """True while ``side`` pinches thumb to index with the other three fingers straight.

        The gripper is toggled on each new pinch. Pinching cannot be confused with the fist clutch
        (a fist curls all four fingers). The clutch hand ignores pinches while the clutch is
        closed, because in a fist the thumb often rests against the index finger.
        """
        cfg = self._cfg
        state = self._pinch_state[side]
        positions = np.asarray(getattr(tv, f"{side}_hand_positions"), dtype=float)
        ratios = finger_ratios(positions)
        tracked = ratios is not None and is_valid_pose(np.asarray(getattr(tv, f"{side}_arm_pose"), dtype=float))
        pinching = False
        if tracked and not (side == self._other and self._clutch_on):
            distance = float(np.linalg.norm(positions[THUMB_TIP_JOINT] - positions[INDEX_TIP_JOINT]))
            limit = cfg.gripper_pinch_off_m if state["on"] else cfg.gripper_pinch_on_m
            others_straight = float(np.mean(ratios[1:])) >= cfg.fist_open_ratio
            pinching = others_straight and distance <= limit
            out.detail += f" | {side} pinch {1e3 * distance:.0f} mm" + ("" if others_straight else " (other fingers curled)")
        if not pinching:
            state["on"], state["since"] = False, None
        else:
            if state["since"] is None:
                state["since"] = now
            if now - state["since"] >= cfg.gripper_pinch_hold_s:
                state["on"] = True
        return state["on"]

    def _fist_clutch(self, tv, now: float, out: OperatorInput) -> float:
        """1.0 while the non-driving hand is a closed fist, with hysteresis and short delays.

        A fist must be held ``clutch_engage_delay_s`` to close the clutch; an open or lost hand
        must persist ``clutch_release_delay_s`` to open it, so single-frame tracking glitches do
        nothing. An untracked hand never counts as a fist.
        """
        cfg = self._cfg
        ratio = fist_ratio(getattr(tv, f"{self._other}_hand_positions"))
        tracked = ratio is not None and is_valid_pose(np.asarray(getattr(tv, f"{self._other}_arm_pose"), dtype=float))
        out.detail = (
            f"{self._other} fist ratio {ratio:.2f} (fist <= {cfg.fist_closed_ratio:.2f}, open >= {cfg.fist_open_ratio:.2f})"
            if tracked
            else f"{self._other} hand NOT tracked: hold it in front of the headset"
        )
        if not self._clutch_on:
            if tracked and ratio <= cfg.fist_closed_ratio:
                if self._clutch_since is None:
                    self._clutch_since = now
                if now - self._clutch_since >= cfg.clutch_engage_delay_s:
                    self._clutch_on, self._clutch_since = True, None
            else:
                self._clutch_since = None
        else:
            if (not tracked) or ratio >= cfg.fist_open_ratio:
                if self._clutch_since is None:
                    self._clutch_since = now
                if now - self._clutch_since >= cfg.clutch_release_delay_s:
                    self._clutch_on, self._clutch_since = False, None
            else:
                self._clutch_since = None
        return 1.0 if self._clutch_on else 0.0

    def diagnostics(self) -> str:
        counts = self._tv.fr10_event_counts
        rejected = {k: v for k, v in counts.items() if k != "accepted" and v}
        text = f"xr events accepted: {counts['accepted']}"
        if rejected:
            text += "; rejected: " + ", ".join(f"{EVENT_REASON_TEXT.get(k, k)} x{v}" for k, v in rejected.items())
        age = time.monotonic() - self._tv.fr10_motion_time if self._tv.fr10_motion_time else None
        text += "; last accepted event: " + ("never" if age is None else f"{age:.1f} s ago")
        return text

    def read(self) -> OperatorInput:
        tv = self._tv
        out = OperatorInput(connected=bool(tv.motion_data_ready))
        if not out.connected:
            return out

        received = tv.fr10_motion_time
        raw = np.asarray(getattr(tv, f"{self._side}_arm_pose"), dtype=float)
        now = time.monotonic()
        fresh = received > 0.0 and 0.0 <= now - received <= self._cfg.stale_timeout_s
        if is_valid_pose(raw):
            out.hand_pose = openxr_to_robot(raw)
            if self._hand_mode:  # tracked hands report wrist axes; controllers already use arm axes
                out.hand_pose = out.hand_pose @ T_ARM_FROM_WRIST[self._side]
            out.valid = fresh
        head = np.asarray(tv.head_pose, dtype=float)
        if is_valid_pose(head):
            out.head_pose = openxr_to_robot(head)

        if self._hand_mode:
            # Clutch: a closed fist of the other hand.
            now = time.monotonic()
            out.clutch = self._fist_clutch(tv, now, out)
            if self._cfg.gripper_source in ("other_hand_pinch", "both_hands_pinch"):
                # Gripper: a pinch (other fingers straight), reported as 0 or 1. The teleoperator
                # toggles the gripper on each new pinch.
                sides = [self._other] if self._cfg.gripper_source == "other_hand_pinch" else [self._other, self._side]
                out.grip = 1.0 if any([self._pinch_gesture(tv, now, out, side) for side in sides]) else 0.0
            elif self._cfg.gripper_source == "driving_hand_fist":
                out.grip = self._fist_grip(tv, now, out)
            else:
                distance = float(getattr(tv, f"{self._side}_hand_pinchValue"))
                span = self._cfg.pinch_open_m - self._cfg.pinch_closed_m
                out.grip = float(np.clip((self._cfg.pinch_open_m - distance) / span, 0.0, 1.0))
        else:
            out.clutch = float(np.clip(getattr(tv, f"{self._side}_ctrl_squeezeValue"), 0.0, 1.0))
            out.grip = float(np.clip(getattr(tv, f"{self._side}_ctrl_triggerValue"), 0.0, 1.0))
            out.button_a = bool(getattr(tv, f"{self._side}_ctrl_aButton"))
            out.button_b = bool(getattr(tv, f"{self._side}_ctrl_bButton"))
        return out

    def close(self) -> None:
        try:
            self._tv.close()
        except Exception as exc:  # noqa: BLE001 - shutting down
            logger.debug("televuer close: %s", exc)


class ScriptedInput(InputSource):
    """Synthetic operator for demos and tests (no headset needed).

    After ``start_delay`` seconds it closes the clutch and repeats: trace a slanted circle,
    yaw the wrist left and right, close and open the gripper.
    """

    def __init__(
        self,
        radius: float = 0.12,
        period: float = 8.0,
        start_delay: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.radius = radius
        self.period = period
        self.start_delay = start_delay
        self._clock = clock
        self._t0: Optional[float] = None
        self._origin = np.array([0.35, -0.20, 1.10])  # a comfortable right-hand position
        self._head = make_pose(np.eye(3), [0.0, 0.0, 1.60])

    def read(self) -> OperatorInput:
        now = self._clock()
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0 - self.start_delay
        if t < 0.0:
            return OperatorInput(True, True, make_pose(PALM_DOWN["right"], self._origin), self._head, 0.0, 0.0)
        phase = 2.0 * math.pi * t / self.period
        offset = self.radius * np.array([math.sin(phase), 1.0 - math.cos(phase), 0.5 * math.sin(phase)])
        # palm down, fingers forward, yawing left and right and nodding a little
        rotation = rot_z(0.4 * math.sin(phase)) @ rot_y(0.2 * math.sin(2.0 * phase)) @ PALM_DOWN["right"]
        grip = 1.0 if (t % self.period) > 0.5 * self.period else 0.0
        return OperatorInput(True, True, make_pose(rotation, self._origin + offset), self._head, 1.0, grip)
