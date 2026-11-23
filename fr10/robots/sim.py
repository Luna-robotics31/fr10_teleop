"""Robot backend for the Isaac Lab simulator (``sim_fr10.py``), linked over ZeroMQ."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Sequence

import numpy as np

from ..comm import ArmCommand, ArmState, SimClientLink
from ..config import Config
from .base import Robot, RobotError, RobotState

logger = logging.getLogger(__name__)


class SimRobot(Robot):
    name = "sim"

    def __init__(self, cfg: Config, lower: Sequence[float], upper: Sequence[float]):
        super().__init__(cfg, lower, upper)
        self._link: Optional[SimClientLink] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_stop = threading.Event()
        self._state_lock = threading.Lock()
        self._latest: Optional[ArmState] = None
        self._latest_stamp = 0.0
        self._send_lock = threading.Lock()
        self._seq = 0
        self._reset_id = 0

    # ------------------------------------------------------------------ Robot hooks
    def _connect(self) -> None:
        sim = self.cfg.sim
        self._link = SimClientLink(sim.host, sim.state_port, sim.command_port)
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(target=self._receive_loop, name="sim-state", daemon=True)
        self._rx_thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._state_lock:
                if self._latest is not None:
                    # Continue the simulator's reset counter so that a restarted teleop
                    # program does not trigger a scene reset by accident.
                    self._reset_id = self._latest.reset_id
                    return
            time.sleep(0.02)
        bad = self._link.bad_messages
        self._disconnect()
        hint = f" ({bad} incompatible messages received: update both programs)" if bad else ""
        raise RobotError(
            f"no state from the simulator at {sim.host}:{sim.state_port}{hint}. "
            "Start it first:  conda activate env_isaaclab && python sim_fr10.py"
        )

    def _disconnect(self) -> None:
        self._rx_stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._link is not None:
            self._link.close()
            self._link = None

    def _receive_loop(self) -> None:
        link = self._link
        while not self._rx_stop.is_set():
            state = link.receive_state(timeout_ms=100)
            if state is None:
                continue
            with self._state_lock:
                self._latest = state
                self._latest_stamp = time.monotonic()

    def _read_state(self) -> RobotState:
        with self._state_lock:
            latest, stamp = self._latest, self._latest_stamp
        if latest is None:
            raise RobotError("sim: no state received yet")
        return RobotState(
            stamp=stamp,
            q=np.asarray(latest.q, dtype=float),
            dq=np.asarray(latest.dq, dtype=float),
            gripper=latest.gripper,
            info={
                "sim_time": latest.sim_time,
                "objects": latest.objects,
                "reset_id": latest.reset_id,
                "cmd_seq": latest.cmd_seq,
            },
        )

    def _send_setpoint(self, q: Optional[np.ndarray], gripper: Optional[float]) -> None:
        with self._send_lock:
            self._seq += 1
            command = ArmCommand(
                seq=self._seq,
                stamp=time.time(),
                q=None if q is None else q.tolist(),
                gripper=gripper,
                reset_id=self._reset_id,
            )
            self._link.send_command(command)

    def view_yaw(self) -> Optional[float]:
        with self._state_lock:
            return None if self._latest is None else self._latest.view_yaw

    # ------------------------------------------------------------------ scene reset
    def reset_scene(self, timeout: float = 5.0) -> bool:
        """Ask the simulator to restore the initial scene and wait until it has done so.

        Set-points are paused during the reset and resume from the new joint position, so the
        arm does not jump back to where it was.
        """
        if self._link is None:
            raise RobotError("sim: not connected")
        with self._lock:
            self._hold = True
            # Reset restores an open gripper as well as the arm's home pose. Held
            # servo packets must not reapply the pre-reset closing command.
            self._gripper_command = 0.0
        try:
            self._reset_id += 1
            wanted = self._reset_id
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self.streaming:  # nobody else is sending: push the request ourselves
                    self._send_setpoint(None, None)
                with self._state_lock:
                    done = self._latest is not None and self._latest.reset_id == wanted
                if done:
                    time.sleep(0.1)  # let the simulator settle and publish post-reset joints
                    self.resync()
                    logger.info("[sim] scene reset")
                    return True
                time.sleep(0.02)
            raise RobotError("sim: the simulator did not confirm the scene reset")
        finally:
            with self._lock:
                self._hold = False
