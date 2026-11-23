"""In-process stand-in for the robot: no simulator, no hardware.

The joints follow the commanded set-points through a first-order lag, which is close to how
a position-controlled industrial arm behaves. Use it to check headset mapping or to run
the test-suite without a simulator or physical robot.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional, Sequence

import numpy as np

from ..config import Config
from .base import Robot, RobotState


class MockRobot(Robot):
    name = "mock"

    def __init__(
        self,
        cfg: Config,
        lower: Sequence[float],
        upper: Sequence[float],
        initial_q: Optional[Sequence[float]] = None,
        lag_s: float = 0.03,
        gripper_stroke_s: float = 0.5,
    ):
        super().__init__(cfg, lower, upper)
        self._state_lock = threading.Lock()
        self._initial_q = np.asarray(cfg.robot.home_rad if initial_q is None else initial_q, dtype=float)
        self._q = self._initial_q.copy()
        self._dq = np.zeros(6)
        self._gripper = 0.0
        self._lag = float(lag_s)
        self._gripper_rate = 1.0 / float(gripper_stroke_s)
        self._injected_fault: Optional[str] = None
        self._frozen = False
        self._stamp = time.monotonic()
        self.setpoints_received = 0

    # ------------------------------------------------------------------ test hooks
    def inject_fault(self, message: Optional[str]) -> None:
        """Make the robot report (or stop reporting) a fault."""
        self._injected_fault = message

    def freeze_feedback(self, frozen: bool) -> None:
        """Stop refreshing the feedback time stamp, as a dead connection would."""
        self._frozen = frozen

    def set_joint_positions(self, q: Sequence[float]) -> None:
        """Teleport the arm (as if someone moved it by hand)."""
        with self._state_lock:
            self._q = np.asarray(q, dtype=float).copy()
            self._dq = np.zeros(6)

    # ------------------------------------------------------------------ Robot hooks
    def _connect(self) -> None:
        self._stamp = time.monotonic()

    def _disconnect(self) -> None:
        pass

    def _read_state(self) -> RobotState:
        with self._state_lock:
            if not self._frozen:
                self._stamp = time.monotonic()
            return RobotState(
                stamp=self._stamp,
                q=self._q.copy(),
                dq=self._dq.copy(),
                gripper=self._gripper,
                fault=self._injected_fault,
            )

    def _send_setpoint(self, q: Optional[np.ndarray], gripper: float) -> None:
        dt = self.servo_period
        with self._state_lock:
            self.setpoints_received += 1
            if q is not None and self._injected_fault is None:
                alpha = 1.0 if self._lag <= 0.0 else 1.0 - math.exp(-dt / self._lag)
                new_q = self._q + alpha * (q - self._q)
                self._dq = (new_q - self._q) / dt
                self._q = new_q
            else:
                self._dq = np.zeros(6)
            step = self._gripper_rate * dt
            self._gripper += float(np.clip(gripper - self._gripper, -step, step))

    def _clear_fault(self) -> None:
        self._injected_fault = None

    def reset_scene(self) -> bool:
        self.set_joint_positions(self._initial_q)
        self.resync()
        return True
