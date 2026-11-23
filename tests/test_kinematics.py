"""Unit tests: FR10 model and inverse kinematics."""

import math
import unittest

import numpy as np

from fr10.config import Config
from fr10.kinematics import IKSolver, RobotModel
from fr10.math_utils import exp_so3, make_pose, pose_error, rot_z

# FR10 link lengths from the FAIRINO DH table [m]
D1, A2, A3, D4, D5, D6 = 0.180, 0.700, 0.586, 0.159, 0.114, 0.106


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config().validate()
        cls.model = RobotModel(cls.cfg.robot)

    def test_structure(self):
        self.assertEqual(self.model.nq, 6)
        self.assertEqual(self.model.joint_names, ["j1", "j2", "j3", "j4", "j5", "j6"])
        self.assertTrue(np.all(self.model.lower < self.model.upper))
        self.assertTrue(self.model.within_limits(self.model.home))

    def test_zero_pose_matches_dh_table(self):
        cfg = Config()
        cfg.robot.tcp_offset = [0.0] * 6  # TCP == flange
        flange = RobotModel(cfg.robot).fk(np.zeros(6))
        np.testing.assert_allclose(flange[:3, 3], [-(A2 + A3), -(D4 + D6), D1 - D5], atol=1e-4)
        np.testing.assert_allclose(flange[:3, 2], [0.0, -1.0, 0.0], atol=1e-4)  # tool axis

    def test_home_pose_is_tool_down_above_table(self):
        tcp = self.model.fk(self.model.home)
        np.testing.assert_allclose(tcp[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)
        np.testing.assert_allclose(tcp[:3, 3], [0.664, 0.002, 0.396], atol=2e-3)
        self.assertGreater(self.model.manipulability(self.model.home), 0.2)

    def test_tcp_offset_is_applied_along_tool_axis(self):
        cfg = Config()
        cfg.robot.tcp_offset = [0.0] * 6
        flange = RobotModel(cfg.robot).fk(self.model.home)
        tcp = self.model.fk(self.model.home)
        np.testing.assert_allclose(tcp[:3, 3] - flange[:3, 3], 0.130 * flange[:3, 2], atol=1e-9)
        np.testing.assert_allclose(tcp[:3, :3], flange[:3, :3], atol=1e-9)

    def test_jacobian_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            q = rng.uniform(self.model.lower, self.model.upper)
            jac = self.model.jacobian(q)
            base = self.model.fk(q)
            for i in range(6):
                dq = np.zeros(6)
                dq[i] = 1e-6
                moved = self.model.fk(q + dq)
                np.testing.assert_allclose((moved[:3, 3] - base[:3, 3]) / 1e-6, jac[:3, i], atol=1e-4)

    def test_bad_configuration_is_rejected(self):
        cfg = Config()
        cfg.robot.home_deg = [0.0, 120.0, 0.0, 0.0, 0.0, 0.0]  # j2 beyond its limit
        with self.assertRaises(ValueError):
            RobotModel(cfg.robot)
        cfg = Config()
        cfg.robot.arm_joints = ["j1", "j2", "j3", "j4", "j5", "nope"]
        with self.assertRaises(ValueError):
            RobotModel(cfg.robot)
        cfg = Config()
        cfg.robot.urdf = "assets/does_not_exist.urdf"
        with self.assertRaises(FileNotFoundError):
            RobotModel(cfg.robot)


class IKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config().validate()
        cls.model = RobotModel(cls.cfg.robot)
        cls.ik = IKSolver(cls.model, cls.cfg.ik)
        cls.max_step = math.radians(cls.cfg.ik.max_step_deg)

    def test_holds_still_at_current_pose(self):
        q = self.model.home
        result = self.ik.solve(self.model.fk(q), q)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.q, q, atol=2e-4)

    def test_zero_orientation_error_is_not_a_special_case(self):
        """Pure translations and exact 'stay here' targets must never upset the solver.

        (A matrix-logarithm orientation error has an undefined gradient exactly at zero error.)
        """
        failures_before = self.ik.failures
        rng = np.random.default_rng(7)
        q = self.model.home.copy()
        for _ in range(300):
            target = self.model.fk(q)  # exactly the current pose
            result = self.ik.solve(target, q)
            self.assertTrue(result.success, result.message)
            np.testing.assert_allclose(result.q, q, atol=1e-4)
        tcp = self.model.fk(q)
        for k in range(600):
            target = tcp.copy()  # orientation held exactly, position moving
            target[:3, 3] += 0.15 * np.array([math.sin(0.02 * k), math.sin(0.031 * k), 0.5 * math.sin(0.017 * k)])
            result = self.ik.solve(target, q)
            self.assertTrue(result.success, result.message)
            self.assertLess(result.position_error, 1e-3)
            self.assertLess(result.rotation_error, 2e-3)
            q = result.q
        for _ in range(200):  # exact identity / axis-aligned rotations as targets
            target = make_pose(np.eye(3), self.model.fk(q)[:3, 3] + rng.uniform(-0.002, 0.002, 3))
            self.assertTrue(np.all(np.isfinite(self.ik.solve(target, q).q)))
        self.assertEqual(self.ik.failures, failures_before)

    def test_tracks_small_displacements_accurately(self):
        rng = np.random.default_rng(2)
        q = self.model.home.copy()
        home_tcp = self.model.fk(q)
        for _ in range(100):
            target = home_tcp.copy()
            target[:3, 3] += rng.uniform(-0.01, 0.01, 3)
            target[:3, :3] = exp_so3(rng.uniform(-0.03, 0.03, 3)) @ home_tcp[:3, :3]
            result = self.ik.solve(target, q)
            self.assertTrue(result.success)
            self.assertLess(result.position_error, 5e-4)
            self.assertLess(result.rotation_error, 5e-3)
            self.assertLessEqual(np.max(np.abs(result.q - q)), self.max_step + 1e-9)
            q = result.q

    def test_follows_a_cartesian_path_smoothly(self):
        """Circle of 0.2 m radius in front of the robot at 60 Hz, 0.25 m/s."""
        dt = 1.0 / 60.0
        q = self.model.home.copy()
        centre = self.model.fk(q)
        worst_position, worst_step, times = 0.0, 0.0, []
        steps = int(2 * math.pi * 0.2 / 0.25 / dt)
        for k in range(steps):
            angle = 2 * math.pi * k / steps
            target = centre.copy()
            target[:3, 3] += [0.2 * math.sin(angle), 0.2 * (1 - math.cos(angle)), 0.1 * math.sin(2 * angle)]
            target[:3, :3] = rot_z(0.5 * math.sin(angle)) @ centre[:3, :3]
            result = self.ik.solve(target, q)
            self.assertTrue(result.success, result.message)
            worst_position = max(worst_position, result.position_error)
            worst_step = max(worst_step, float(np.max(np.abs(result.q - q))))
            times.append(result.solve_time)
            q = result.q
        self.assertLess(worst_position, 1e-3)
        self.assertLess(worst_step, math.radians(3.0))  # no joint jumps
        self.assertLess(float(np.median(times)), 0.010)  # fast enough for a 60 Hz loop

    def test_solution_stays_inside_joint_limits(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            q_seed = rng.uniform(self.model.lower, self.model.upper)
            target = make_pose(exp_so3(rng.uniform(-3, 3, 3)), rng.uniform(-1.5, 1.5, 3))
            result = self.ik.solve(target, q_seed)
            self.assertTrue(np.all(np.isfinite(result.q)))
            self.assertTrue(self.model.within_limits(result.q, 1e-9))
            self.assertLessEqual(np.max(np.abs(result.q - q_seed)), self.max_step + 1e-9)

    def test_seed_outside_limits_is_clamped(self):
        q_seed = self.model.upper + 0.5
        result = self.ik.solve(self.model.fk(self.model.home), q_seed)
        self.assertTrue(self.model.within_limits(result.q, 1e-9))

    def test_unreachable_target_degrades_gracefully(self):
        q = self.model.home.copy()
        target = self.model.fk(q)
        target[:3, 3] = [3.0, 0.0, 0.5]  # far outside the 1.4 m reach
        initial_error = pose_error(self.model.fk(q), target)[0]
        for _ in range(60):
            result = self.ik.solve(target, q)
            self.assertTrue(self.model.within_limits(result.q, 1e-9))
            self.assertLessEqual(np.max(np.abs(result.q - q)), self.max_step + 1e-9)
            q = result.q
        final = pose_error(self.model.fk(q), target)[0]
        self.assertLess(final, initial_error - 0.5)  # moved at least 0.5 m towards it
        self.assertLess(np.linalg.norm(self.model.fk(q)[:3, 3]), 1.75)  # arm stretched towards it
        # and it comes back when the target returns
        back = self.ik.solve_global(self.model.fk(self.model.home), q, tolerance=1e-3)
        self.assertIsNotNone(back)

    def test_passes_through_wrist_singularity_without_jumping(self):
        """Tilt the tool until j5 crosses zero while following the target at 60 Hz."""
        q = self.model.home.copy()
        start = self.model.fk(q)
        worst_step = 0.0
        for k in range(240):
            tilt = math.radians(140.0) * min(1.0, k / 180.0)
            target = start.copy()
            # rotate the tool about its own x-axis: brings joint 5 through 0
            target[:3, :3] = start[:3, :3] @ exp_so3([tilt, 0.0, 0.0])
            result = self.ik.solve(target, q)
            worst_step = max(worst_step, float(np.max(np.abs(result.q - q))))
            self.assertTrue(np.all(np.isfinite(result.q)))
            q = result.q
        self.assertLessEqual(worst_step, self.max_step + 1e-9)

    def test_solve_global_reaches_distant_pose(self):
        target = make_pose(np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]), [0.6, 0.2, 0.25])
        q = self.ik.solve_global(target, self.model.home, tolerance=1e-3)
        self.assertIsNotNone(q)
        position_error, rotation_error = pose_error(self.model.fk(q), target)
        self.assertLess(position_error, 1e-3)
        self.assertLess(rotation_error, 1e-2)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            self.ik.solve(np.full((4, 4), np.nan), self.model.home)
        with self.assertRaises(ValueError):
            self.ik.solve(np.eye(3), self.model.home)


if __name__ == "__main__":
    unittest.main()
