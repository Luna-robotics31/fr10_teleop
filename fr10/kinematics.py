"""FR10 kinematic model and inverse kinematics.

The model is built from the same URDF the simulator uses. Gripper joints are locked, leaving
the six arm joints, and a ``tcp`` frame is attached to the flange with the configured offset.

Inverse kinematics is posed as a small non-linear program (Pinocchio + CasADi + IPOPT, the
approach used by the reference xr_teleoperate project):

    minimise   w_p |p(q) - p*|^2 + w_r/2 |R(q) - R*|_F^2
             + w_n |q - q_home|^2 + w_s |q - q_prev|^2
    subject to q_min <= q <= q_max   and   |q - q_prev| <= max_step

The step bound keeps every solution in the neighbourhood of the previous one, so the arm can
never jump to a different elbow/wrist configuration between two control cycles.

The orientation term is the squared Frobenius distance between the rotation matrices,
``|R - R*|_F^2 / 2 = 2 (1 - cos(angle))``, which equals ``angle^2`` for small errors. Unlike
the matrix logarithm it is a polynomial in the entries of ``R``: its derivatives are finite
everywhere, in particular at zero error, which is where a teleoperated arm spends most of
its time (clutch just closed, orientation locked, hand held still).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pinocchio as pin

from .config import IKConfig, RobotConfig
from .math_utils import pose_error, xyzrpy_to_pose

logger = logging.getLogger(__name__)

TCP_FRAME = "tcp"


class RobotModel:
    """Six-axis kinematic model of the FR10 with a tool-centre-point frame."""

    def __init__(self, cfg: RobotConfig):
        urdf_path = Path(cfg.urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.urdf_path = urdf_path
        self.joint_names = list(cfg.arm_joints)

        full_model = pin.buildModelFromUrdf(str(urdf_path))
        missing = [name for name in self.joint_names if not full_model.existJointName(name)]
        if missing:
            raise ValueError(f"joints {missing} are not in {urdf_path.name}")
        if not full_model.existFrame(cfg.flange_frame):
            raise ValueError(f"frame '{cfg.flange_frame}' is not in {urdf_path.name}")

        # Lock everything that is not an arm joint (gripper fingers) at its neutral position.
        locked = [
            full_model.getJointId(name)
            for name in full_model.names[1:]  # index 0 is the universe
            if name not in self.joint_names
        ]
        self.model = pin.buildReducedModel(full_model, locked, pin.neutral(full_model))
        if self.model.nq != len(self.joint_names) or self.model.nv != len(self.joint_names):
            raise ValueError("the arm joints must be single-degree-of-freedom joints")
        order = [self.model.names[i] for i in range(1, self.model.njoints)]
        if order != self.joint_names:
            raise ValueError(f"robot.arm_joints must follow the URDF chain order {order}")

        flange = self.model.frames[self.model.getFrameId(cfg.flange_frame)]
        offset = xyzrpy_to_pose(cfg.tcp_offset_xyzrpy)
        tcp_placement = flange.placement * pin.SE3(offset[:3, :3], offset[:3, 3])
        self.model.addFrame(pin.Frame(TCP_FRAME, flange.parentJoint, tcp_placement, pin.FrameType.OP_FRAME))
        self.tcp_frame_id = self.model.getFrameId(TCP_FRAME)
        self.data = self.model.createData()

        margin = np.deg2rad(cfg.joint_limit_margin_deg)
        self.lower = self.model.lowerPositionLimit.copy() + margin
        self.upper = self.model.upperPositionLimit.copy() - margin
        if np.any(self.lower >= self.upper):
            raise ValueError("robot.joint_limit_margin_deg leaves no joint range")
        self.urdf_velocity_limit = self.model.velocityLimit.copy()
        self.home = np.asarray(cfg.home_rad, dtype=float)
        if not self.within_limits(self.home):
            raise ValueError("robot.home_deg violates the joint limits")

    @property
    def nq(self) -> int:
        return self.model.nq

    def within_limits(self, q: Sequence[float], tolerance: float = 0.0) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.lower - tolerance) and np.all(q <= self.upper + tolerance))

    def clamp(self, q: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Pose of the tool centre point in the robot base frame (4x4)."""
        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        placement = pin.updateFramePlacement(self.model, self.data, self.tcp_frame_id)
        return placement.homogeneous.copy()

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """6x6 TCP Jacobian, linear rows first, expressed in base-frame axes."""
        q = np.asarray(q, dtype=float)
        return pin.computeFrameJacobian(
            self.model, self.data, q, self.tcp_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).copy()

    def manipulability(self, q: Sequence[float]) -> float:
        """Smallest singular value of the Jacobian (0 at a singularity)."""
        return float(np.linalg.svd(self.jacobian(q), compute_uv=False)[-1])


@dataclass
class IKResult:
    q: np.ndarray  # solution (always within limits and within max_step of the seed)
    success: bool  # solver converged
    position_error: float  # [m] remaining distance to the target
    rotation_error: float  # [rad]
    solve_time: float  # [s]
    message: str = ""


class IKSolver:
    """Warm-started, joint-limit aware inverse kinematics for teleoperation."""

    def __init__(self, model: RobotModel, cfg: IKConfig):
        import casadi
        from pinocchio import casadi as cpin

        self._casadi = casadi
        self.robot = model
        self.cfg = cfg
        self.max_step = float(np.deg2rad(cfg.max_step_deg))
        nq = model.nq

        cmodel = cpin.Model(model.model)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", nq, 1)
        ctarget = casadi.SX.sym("target", 4, 4)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        tcp = cdata.oMf[model.tcp_frame_id]
        position_error = casadi.Function("position_error", [cq, ctarget], [tcp.translation - ctarget[:3, 3]])
        rotation_error = casadi.Function(
            "rotation_error", [cq, ctarget], [casadi.reshape(tcp.rotation - ctarget[:3, :3], 9, 1)]
        )

        opti = casadi.Opti()
        self._q = opti.variable(nq)
        self._target = opti.parameter(4, 4)
        self._q_prev = opti.parameter(nq)
        self._lower = opti.parameter(nq)
        self._upper = opti.parameter(nq)
        cost = (
            cfg.position_weight * casadi.sumsqr(position_error(self._q, self._target))
            + 0.5 * cfg.rotation_weight * casadi.sumsqr(rotation_error(self._q, self._target))
            + cfg.posture_weight * casadi.sumsqr(self._q - model.home)
            + cfg.smoothness_weight * casadi.sumsqr(self._q - self._q_prev)
        )
        opti.minimize(cost)
        opti.subject_to(opti.bounded(self._lower, self._q, self._upper))
        opti.solver(
            "ipopt",
            {
                "expand": True,
                "detect_simple_bounds": True,
                "calc_lam_p": False,
                "print_time": False,
                "ipopt.sb": "yes",
                "ipopt.print_level": 0,
                "ipopt.max_iter": int(cfg.max_iterations),
                "ipopt.tol": float(cfg.tolerance),
                "ipopt.acceptable_tol": float(cfg.tolerance) * 100.0,
                "ipopt.acceptable_iter": 3,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.mu_init": 1e-3,
            },
        )
        self._opti = opti
        self._cost = casadi.Function("cost", [self._q, self._target, self._q_prev], [cost])
        self.failures = 0
        self.solves = 0

    def solve(self, target: np.ndarray, q_seed: Sequence[float]) -> IKResult:
        """Find joint angles that bring the TCP to ``target`` (4x4, base frame).

        ``q_seed`` is the previous solution (or the current joint position). The returned
        configuration differs from it by at most ``ik.max_step_deg`` per joint.
        """
        start = time.perf_counter()
        robot = self.robot
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("IK target must be a finite 4x4 matrix")
        q_prev = robot.clamp(q_seed)
        if not np.all(np.isfinite(q_prev)):
            raise ValueError("IK seed must be finite")
        lower = np.maximum(robot.lower, q_prev - self.max_step)
        upper = np.minimum(robot.upper, q_prev + self.max_step)

        opti = self._opti
        opti.set_initial(self._q, q_prev)
        opti.set_value(self._target, target)
        opti.set_value(self._q_prev, q_prev)
        opti.set_value(self._lower, lower)
        opti.set_value(self._upper, upper)

        self.solves += 1
        success, message = True, ""
        try:
            solution = opti.solve()
            q = np.asarray(solution.value(self._q), dtype=float).reshape(-1)
        except RuntimeError as exc:  # IPOPT did not converge: use its last iterate if it is an improvement
            success, message = False, str(exc).strip().splitlines()[-1][:160]
            q = np.asarray(opti.debug.value(self._q), dtype=float).reshape(-1)

        q = np.clip(q, lower, upper) if np.all(np.isfinite(q)) else q_prev
        if not success:
            self.failures += 1
            cost_new = float(self._cost(q, target, q_prev))
            cost_old = float(self._cost(q_prev, target, q_prev))
            if not np.isfinite(cost_new) or cost_new > cost_old:
                q = q_prev

        position_error, rotation_error = pose_error(robot.fk(q), target)
        return IKResult(q, success, position_error, rotation_error, time.perf_counter() - start, message)

    def solve_global(
        self, target: np.ndarray, q_seed: Sequence[float], tolerance: float = 1e-4, max_solves: int = 200
    ) -> Optional[np.ndarray]:
        """Iterate :meth:`solve` until the target is reached (for start-up moves, not for the loop).

        Returns ``None`` if the target cannot be reached from ``q_seed``.
        """
        q = self.robot.clamp(q_seed)
        for _ in range(max_solves):
            result = self.solve(target, q)
            moved = float(np.max(np.abs(result.q - q)))
            q = result.q
            if result.position_error < tolerance and result.rotation_error < 10.0 * tolerance:
                return q
            if moved < 1e-7:
                break
        return None
