"""Operator hand motion -> tool-centre-point target (relative, clutched teleoperation).

While the clutch is closed, the tool follows the *change* of the operator's hand pose
since the moment the clutch closed::

    p_target = p_tcp0 + scale * S (p_hand - p_hand0)
    R_target = S (R_hand R_hand0^T) S^T  R_tcp0

``S`` rotates operator directions into the robot base frame. Because only pose changes
are used, the operator can stand anywhere, re-centre by releasing and re-closing the
clutch, and no calibration between headset and robot is needed.

All hand/head poses given to this module use the "robot" axis convention in the headset's
world frame: x forward, y left, z up (see :mod:`fr10.xr_input`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import MappingConfig, WorkspaceConfig
from .math_utils import (
    exp_so3,
    is_valid_pose,
    log_so3,
    make_pose,
    orthonormalize,
    rot_z,
    rotation_angle,
    rpy_to_matrix,
    slerp_rotation,
    yaw_of,
)

# Height of the FR10 shoulder joint (j2 axis) above the base plane [m].
SHOULDER_HEIGHT = 0.18

# Orientation of a hand/controller pose when the palm faces down and the fingers point forward,
# in robot axes (x forward, y left, z up). Hand poses follow the Unitree arm convention that
# televuer uses: x along the fingers, and (right hand) y from the back of the hand to the palm,
# z from the little finger to the index finger; (left hand) y from the palm to the back of the hand.
PALM_DOWN = {
    "right": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "left": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}


class Workspace:
    """Region the tool centre point may be commanded to (robot base frame).

    Intersection of an axis-aligned box, a sphere around the shoulder (maximum reach) and the
    outside of a vertical cylinder around the base axis (keeps the tool away from the robot's
    own column).
    """

    def __init__(self, cfg: WorkspaceConfig):
        self.lo = np.array([cfg.x[0], cfg.y[0], cfg.z[0]], dtype=float)
        self.hi = np.array([cfg.x[1], cfg.y[1], cfg.z[1]], dtype=float)
        self.max_reach = float(cfg.max_reach)
        self.min_radius = float(cfg.min_radius)
        self.shoulder = np.array([0.0, 0.0, SHOULDER_HEIGHT])
        self.anchor = self._find_anchor()

    def _margin(self, p: np.ndarray) -> float:
        """Distance to the nearest limit; negative outside the workspace."""
        return float(
            min(
                np.min(p - self.lo),
                np.min(self.hi - p),
                self.max_reach - np.linalg.norm(p - self.shoulder),
                np.hypot(p[0], p[1]) - self.min_radius,
            )
        )

    def _find_anchor(self) -> np.ndarray:
        """A point well inside the workspace (used to resolve corner cases of :meth:`clamp`)."""
        grid = [np.linspace(self.lo[i], self.hi[i], 9) for i in range(3)]
        candidates = np.array(np.meshgrid(*grid, indexing="ij")).reshape(3, -1).T
        margins = np.array([self._margin(c) for c in candidates])
        best = int(np.argmax(margins))
        if margins[best] <= 0.0:
            raise ValueError("mapping.workspace is empty: box, max_reach and min_radius do not overlap")
        return candidates[best]

    def contains(self, position, tolerance: float = 1e-9) -> bool:
        return self._margin(np.asarray(position, dtype=float)) >= -tolerance

    def clamp(self, position) -> np.ndarray:
        """Closest allowed position (exact for the box/reach limits, always inside the workspace)."""
        p = np.asarray(position, dtype=float)
        if self.contains(p, 0.0):
            return p.copy()
        q = np.clip(p, self.lo, self.hi)
        arm = q - self.shoulder
        reach = float(np.linalg.norm(arm))
        if reach > self.max_reach:
            q = self.shoulder + arm * (self.max_reach / reach)
        radius = float(np.hypot(q[0], q[1]))
        if radius < self.min_radius:
            direction = q[:2] / radius if radius > 1e-6 else np.array([1.0, 0.0])
            q[:2] = direction * self.min_radius
        q = np.clip(q, self.lo, self.hi)
        if self.contains(q):
            return q
        # The sequential projection left one limit violated (only near edges where several
        # limits meet): walk back towards the anchor until every limit holds.
        inside, outside = 0.0, 1.0
        for _ in range(48):
            mid = 0.5 * (inside + outside)
            if self.contains(self.anchor + mid * (q - self.anchor), 0.0):
                inside = mid
            else:
                outside = mid
        return self.anchor + inside * (q - self.anchor)


@dataclass
class MappingResult:
    target: np.ndarray  # 4x4 TCP target in the robot base frame (smoothed, limited)
    limited: bool  # the workspace limits changed the requested position
    jump: bool = False  # tracking jump detected; the clutch has been released


class WorkspaceMapper:
    """Stateful clutch + mapping from hand pose to TCP target."""

    def __init__(self, cfg: MappingConfig, hand: str = "right", neutral_rotation: Optional[np.ndarray] = None):
        """
        Args:
            cfg: mapping configuration.
            hand: which hand drives the arm ("right" or "left"); sets the palm-down reference.
            neutral_rotation: tool orientation that "palm down, fingers forward" must produce
                (absolute mode). Default: gripper pointing straight down, jaws opening sideways,
                which is the orientation of the default home pose.
        """
        if hand not in PALM_DOWN:
            raise ValueError("hand must be 'right' or 'left'")
        self.cfg = cfg
        self.workspace = Workspace(cfg.workspace)
        self.hand = hand
        neutral = (
            np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            if neutral_rotation is None
            else orthonormalize(np.asarray(neutral_rotation, dtype=float)[:3, :3])
        )
        self._neutral = neutral
        offset = rpy_to_matrix(*np.deg2rad(cfg.hand_offset_rpy_deg))
        # tool orientation = S R_hand (offset PALM_DOWN^T neutral): palm down gives `neutral`,
        # and turning the hand about any world axis turns the tool about the same axis.
        self._hand_to_tool = offset @ PALM_DOWN[hand].T @ neutral
        self._engaged = False
        self._has_target = False
        self._operator_to_base = np.eye(3)
        self._hand0_p = np.zeros(3)
        self._hand0_R = np.eye(3)
        self._tcp0_p = np.zeros(3)
        self._tcp0_R = np.eye(3)
        self._last_hand_p = np.zeros(3)
        self._last_hand_R = np.eye(3)
        self._out_p = np.zeros(3)
        self._out_R = np.eye(3)
        self._prev_request_p = np.zeros(3)
        self._prev_request_R = np.eye(3)
        self._last_head_yaw = 0.0

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def target(self) -> Optional[np.ndarray]:
        """Latest commanded TCP pose, or ``None`` if the clutch was never closed."""
        return make_pose(self._out_R, self._out_p) if self._has_target else None

    def _operator_to_base_rotation(self, head_pose: Optional[np.ndarray], yaw_offset: float) -> tuple:
        """Rotation taking headset-world directions to robot directions, and the heading used."""
        operator_yaw = 0.0
        if self.cfg.reference == "head_yaw":
            operator_yaw = self._last_head_yaw
            if head_pose is not None and is_valid_pose(head_pose):
                operator_yaw = yaw_of(head_pose[:3, :3], fallback=operator_yaw)
        return rot_z(yaw_offset) @ rot_z(operator_yaw).T, operator_yaw

    def absolute_tool_rotation(self, hand_rotation: np.ndarray, operator_to_base: np.ndarray) -> np.ndarray:
        """Tool orientation that corresponds to a hand orientation (absolute mode, unscaled)."""
        return operator_to_base @ orthonormalize(hand_rotation) @ self._hand_to_tool

    def alignment_error(
        self, hand_pose: np.ndarray, head_pose: Optional[np.ndarray], tcp_pose: np.ndarray, yaw_offset: float
    ) -> float:
        """Angle [deg] between the tool and the orientation the hand asks for.

        Always 0 unless orientation follows the hand in absolute mode. The clutch should only
        close when this is below ``mapping.engage_tolerance_deg``.
        """
        cfg = self.cfg
        if cfg.orientation_mode != "absolute" or not cfg.track_orientation or cfg.rotation_scale <= 0.0:
            return 0.0
        if not is_valid_pose(hand_pose) or not is_valid_pose(tcp_pose):
            return 180.0
        to_base, _ = self._operator_to_base_rotation(head_pose, yaw_offset)
        wanted = self._scaled_absolute(self.absolute_tool_rotation(hand_pose[:3, :3], to_base))
        return math.degrees(rotation_angle(orthonormalize(tcp_pose[:3, :3]).T @ wanted))

    def _scaled_absolute(self, tool_rotation: np.ndarray) -> np.ndarray:
        scale = self.cfg.rotation_scale
        if scale == 1.0:
            return tool_rotation
        return exp_so3(scale * log_so3(tool_rotation @ self._neutral.T)) @ self._neutral

    def engage(self, hand_pose: np.ndarray, head_pose: Optional[np.ndarray], tcp_pose: np.ndarray, yaw_offset: float) -> None:
        """Close the clutch.

        Args:
            hand_pose: current hand/controller pose (headset world frame, robot axis convention).
            head_pose: current head pose in the same frame (may be ``None`` or invalid).
            tcp_pose: current commanded TCP pose in the robot base frame.
            yaw_offset: rotation about vertical from the operator's forward direction to robot +x [rad].
        """
        if not is_valid_pose(hand_pose) or not is_valid_pose(tcp_pose):
            raise ValueError("cannot engage with an invalid hand or TCP pose")
        self._operator_to_base, operator_yaw = self._operator_to_base_rotation(head_pose, yaw_offset)
        if self.cfg.reference == "head_yaw":
            self._last_head_yaw = operator_yaw

        self._hand0_p = hand_pose[:3, 3].copy()
        self._hand0_R = orthonormalize(hand_pose[:3, :3])
        self._last_hand_p = self._hand0_p.copy()
        self._last_hand_R = self._hand0_R.copy()
        self._tcp0_p = tcp_pose[:3, 3].copy()
        self._tcp0_R = orthonormalize(tcp_pose[:3, :3])
        self._out_p = self._tcp0_p.copy()
        self._out_R = self._tcp0_R.copy()
        self._prev_request_p = self._tcp0_p.copy()
        self._prev_request_R = self._tcp0_R.copy()
        self._engaged = True
        self._has_target = True

    @property
    def hand_travel(self) -> float:
        """Distance [m] the hand has moved since the clutch closed."""
        return float(np.linalg.norm(self._last_hand_p - self._hand0_p))

    @property
    def target_travel(self) -> float:
        """Distance [m] the commanded tool position has moved since the clutch closed."""
        return float(np.linalg.norm(self._out_p - self._tcp0_p))

    def release(self) -> None:
        """Open the clutch; the tool stays at its last target."""
        self._engaged = False

    def update(self, hand_pose: np.ndarray, dt: float) -> MappingResult:
        """Compute the TCP target for the current hand pose. Requires a closed clutch."""
        if not self._engaged:
            raise RuntimeError("WorkspaceMapper.update() called while the clutch is open")
        cfg = self.cfg
        dt = max(float(dt), 0.0)
        hold = MappingResult(make_pose(self._out_R, self._out_p), limited=False, jump=True)
        if not is_valid_pose(hand_pose):
            self._engaged = False
            return hold
        hand_p = hand_pose[:3, 3]
        hand_R = orthonormalize(hand_pose[:3, :3])

        # A controller that loses and regains tracking can teleport: never follow that.
        if (
            np.linalg.norm(hand_p - self._last_hand_p) > cfg.jump_position_m
            or rotation_angle(self._last_hand_R.T @ hand_R) > math.radians(cfg.jump_rotation_deg)
        ):
            self._engaged = False
            return hold
        self._last_hand_p = hand_p.copy()
        self._last_hand_R = hand_R

        # 1) relative hand motion, expressed in the robot base frame
        S = self._operator_to_base
        delta_p = S @ (hand_p - self._hand0_p) * cfg.position_scale
        requested_p = self._tcp0_p + delta_p
        if cfg.track_orientation and cfg.rotation_scale > 0.0 and cfg.orientation_mode == "absolute":
            requested_R = self._scaled_absolute(self.absolute_tool_rotation(hand_R, S))
        elif cfg.track_orientation and cfg.rotation_scale > 0.0:
            delta_R = S @ (hand_R @ self._hand0_R.T) @ S.T
            if cfg.rotation_scale != 1.0:
                delta_R = exp_so3(cfg.rotation_scale * log_so3(delta_R))
            requested_R = delta_R @ self._tcp0_R
        else:
            requested_R = self._tcp0_R

        # 2) workspace limits
        limited_p = self.workspace.clamp(requested_p)
        limited = bool(np.linalg.norm(limited_p - requested_p) > 1e-6)

        # 3) low-pass towards the requested pose; the faster the hand moves, the less it is smoothed
        if cfg.smoothing_time_s <= 0.0 or dt <= 0.0:
            alpha_p = alpha_r = 1.0
        else:
            move_speed = float(np.linalg.norm(limited_p - self._prev_request_p)) / dt
            turn_speed = rotation_angle(self._prev_request_R.T @ requested_R) / dt
            tau_p = cfg.smoothing_time_s / (1.0 + move_speed / cfg.smoothing_fast_speed_m_s)
            tau_r = cfg.smoothing_time_s / (1.0 + turn_speed / math.radians(cfg.smoothing_fast_turn_deg_s))
            alpha_p = 1.0 - math.exp(-dt / tau_p)
            alpha_r = 1.0 - math.exp(-dt / tau_r)
        self._prev_request_p = limited_p.copy()
        self._prev_request_R = requested_R.copy()
        new_p = self._out_p + alpha_p * (limited_p - self._out_p)
        new_R = slerp_rotation(self._out_R, requested_R, alpha_r)

        # 4) speed limits on the target itself
        step = new_p - self._out_p
        step_len = float(np.linalg.norm(step))
        max_step = cfg.max_linear_speed * dt
        if step_len > max_step:
            new_p = self._out_p + step * (max_step / step_len)
        turn = rotation_angle(self._out_R.T @ new_R)
        max_turn = math.radians(cfg.max_angular_speed_deg) * dt
        if turn > max_turn:
            new_R = slerp_rotation(self._out_R, new_R, max_turn / turn)

        self._out_p = new_p
        self._out_R = orthonormalize(new_R)
        return MappingResult(make_pose(self._out_R, self._out_p), limited=limited)
