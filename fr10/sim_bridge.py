"""Simulator-side half of the teleop link.

The physics engine (Isaac Lab in ``sim_fr10.py``, or a trivial kinematic model in the tests)
calls :meth:`SimBridge.poll` before each physics step and :meth:`SimBridge.publish` after
it. All protocol bookkeeping lives here, so every simulator behaves identically towards the
teleoperation program.

Imported by the simulator's conda environment: standard library, NumPy and pyzmq only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .comm import ArmState, SimServerLink
from .config import Config


@dataclass
class BridgeCommand:
    joint_target: np.ndarray  # arm joint targets to apply this step [rad]
    gripper: float  # gripper closing target, 0 open .. 1 closed
    reset: bool  # restore the initial scene before applying the targets
    fresh: bool  # a new command arrived since the previous poll


class SimBridge:
    def __init__(self, cfg: Config, initial_q: Sequence[float], view_yaw: float = 0.0, bind_host: str = "*"):
        self._cfg = cfg
        self._link = SimServerLink(cfg.sim.state_port, cfg.sim.command_port, bind_host)
        self._initial_q = np.asarray(initial_q, dtype=float).copy()
        self._joint_target = self._initial_q.copy()
        self._gripper_target = 0.0
        self._view_yaw = float(view_yaw)
        self._reset_id = 0
        self._cmd_seq = -1
        self._state_seq = 0
        self._last_command_time: Optional[float] = None

    @property
    def joint_target(self) -> np.ndarray:
        return self._joint_target.copy()

    @property
    def gripper_target(self) -> float:
        return self._gripper_target

    @property
    def teleop_connected(self) -> bool:
        """True while commands keep arriving."""
        return (
            self._last_command_time is not None
            and time.monotonic() - self._last_command_time < self._cfg.sim.command_timeout_s
        )

    @property
    def bad_messages(self) -> int:
        return self._link.bad_messages

    def poll(self) -> BridgeCommand:
        """Fetch the newest command (if any) and return the targets to apply."""
        command = self._link.poll_command()
        reset = False
        if command is not None:
            self._last_command_time = time.monotonic()
            self._cmd_seq = command.seq
            if command.reset_id != self._reset_id:
                self._reset_id = command.reset_id
                reset = True
                # After a reset the arm is at its initial pose; hold it there until the
                # teleop program sends targets again (it resynchronises first).
                self._joint_target = self._initial_q.copy()
                self._gripper_target = 0.0
            elif command.q is not None:
                self._joint_target = np.asarray(command.q, dtype=float)
            if not reset and command.gripper is not None:
                self._gripper_target = float(command.gripper)
        return BridgeCommand(self._joint_target.copy(), self._gripper_target, reset, command is not None)

    def publish(
        self,
        sim_time: float,
        q: Sequence[float],
        dq: Sequence[float],
        gripper: float,
        objects: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Send the state measured after the physics step."""
        self._state_seq += 1
        self._link.publish_state(
            ArmState(
                seq=self._state_seq,
                stamp=time.time(),
                sim_time=float(sim_time),
                q=[float(v) for v in q],
                dq=[float(v) for v in dq],
                gripper=float(min(1.0, max(0.0, gripper))),
                cmd_seq=self._cmd_seq,
                reset_id=self._reset_id,
                view_yaw=self._view_yaw,
                objects=objects or {},
            )
        )

    def close(self) -> None:
        self._link.close()
