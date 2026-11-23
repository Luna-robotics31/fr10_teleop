"""Unit tests: math helpers and configuration loading."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from fr10 import math_utils as mu
from fr10.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config, parse_override


def random_rotation(rng) -> np.ndarray:
    return mu.exp_so3(rng.uniform(-1.0, 1.0, 3) * rng.uniform(0.0, np.pi))


class MathTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_exp_log_roundtrip(self):
        for _ in range(200):
            rot = random_rotation(self.rng)
            np.testing.assert_allclose(mu.exp_so3(mu.log_so3(rot)), rot, atol=1e-9)
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)

    def test_log_near_pi_and_zero(self):
        for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0.6, 0.0, 0.8])):
            for angle in (np.pi, np.pi - 1e-7, 1e-10, 0.0):
                rot = mu.exp_so3(axis * angle)
                np.testing.assert_allclose(mu.exp_so3(mu.log_so3(rot)), rot, atol=1e-6)
                self.assertAlmostEqual(np.linalg.norm(mu.log_so3(rot)), angle, places=5)

    def test_quaternion_roundtrip(self):
        for _ in range(200):
            rot = random_rotation(self.rng)
            quat = mu.matrix_to_quat(rot)
            self.assertGreaterEqual(quat[0], 0.0)
            np.testing.assert_allclose(mu.quat_to_matrix(quat), rot, atol=1e-9)

    def test_rpy_matches_urdf_convention(self):
        # URDF: fixed-axis roll, pitch, yaw == Rz(yaw) Ry(pitch) Rx(roll)
        rot = mu.rpy_to_matrix(0.3, -0.2, 1.1)
        np.testing.assert_allclose(rot, mu.rot_z(1.1) @ mu.rot_y(-0.2) @ mu.rot_x(0.3))
        np.testing.assert_allclose(mu.rot_z(math.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_pose_inverse_and_validity(self):
        pose = mu.make_pose(random_rotation(self.rng), [0.1, -0.2, 0.3])
        np.testing.assert_allclose(mu.pose_inverse(pose) @ pose, np.eye(4), atol=1e-12)
        self.assertTrue(mu.is_valid_pose(pose))
        self.assertFalse(mu.is_valid_pose(np.zeros((4, 4))))
        self.assertFalse(mu.is_valid_pose(np.full((4, 4), np.nan)))
        mirrored = pose.copy()
        mirrored[:3, 0] *= -1.0
        self.assertFalse(mu.is_valid_pose(mirrored))

    def test_slerp_and_yaw(self):
        a, b = mu.rot_z(0.2), mu.rot_z(1.0)
        np.testing.assert_allclose(mu.slerp_rotation(a, b, 0.25), mu.rot_z(0.4), atol=1e-12)
        self.assertAlmostEqual(mu.yaw_of(mu.rot_z(0.7) @ mu.rot_y(0.3)), 0.7)
        self.assertEqual(mu.yaw_of(mu.rot_y(-math.pi / 2), fallback=1.23), 1.23)  # looking straight up

    def test_orthonormalize(self):
        rot = random_rotation(self.rng)
        noisy = rot + self.rng.normal(0, 1e-3, (3, 3))
        fixed = mu.orthonormalize(noisy)
        np.testing.assert_allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fixed), 1.0)
        self.assertLess(mu.rotation_angle(rot.T @ fixed), 5e-3)


class ConfigTests(unittest.TestCase):
    def test_default_yaml_matches_dataclass_defaults(self):
        self.assertTrue(DEFAULT_CONFIG_PATH.is_file())
        self.assertEqual(load_config(DEFAULT_CONFIG_PATH).to_dict(), Config().validate().to_dict())

    def test_default_yaml_documents_every_option(self):
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as stream:
            documented = yaml.safe_load(stream)

        def keys(tree, prefix=""):
            out = set()
            for key, value in tree.items():
                out |= keys(value, f"{prefix}{key}.") if isinstance(value, dict) else {prefix + key}
            return out

        self.assertEqual(keys(documented), keys(Config().to_dict()))

    def test_overrides_and_units(self):
        cfg = load_config(overrides={"control.speed_scale": 0.3, "robot": {"joint_limit_margin_deg": 5}})
        self.assertEqual(cfg.control.speed_scale, 0.3)
        self.assertEqual(cfg.robot.joint_limit_margin_deg, 5.0)
        self.assertAlmostEqual(cfg.robot.home_rad[1], math.radians(cfg.robot.home_deg[1]))
        self.assertTrue(cfg.robot.urdf_path.is_file())
        self.assertEqual(parse_override("mapping.yaw_offset_deg=90"), ("mapping.yaw_offset_deg", 90))
        self.assertEqual(parse_override("xr.hand=left"), ("xr.hand", "left"))
        self.assertEqual(load_config(overrides={"mapping.yaw_offset_deg": 90}).mapping.yaw_offset_deg, 90.0)
        self.assertEqual(load_config().mapping.yaw_offset_deg, "auto")

    def test_invalid_values_are_rejected(self):
        bad = [
            {"control.soft_command_lead_deg": 30.0},  # not below the maximum lead
            {"control.max_command_lead_deg": 35.0},  # not below the tracking error limit: a lagging arm would fault

            {"control.speed_scale": 0.0},
            {"control.speed_scale": 1.5},
            {"robot.home_deg": [0, 0, 0]},
            {"xr.hand": "both"},
            {"mapping.reference": "wrist"},
            {"mapping.yaw_offset_deg": "north"},
            {"robot.typo": 1},
            {"nonexistent_section.value": 1},
            {"control.rate_hz": "fast"},
            {"xr.clutch_on": 0.2},
            {"control.rate_hz": 200.0, "control.servo_rate_hz": 100.0},
            {"gripper.enabled": "yes"},
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(ConfigError):
                load_config(overrides=overrides)

    def test_partial_user_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "my.yaml"
            path.write_text("fairino:\n  ip: 10.0.0.7\n  gripper:\n    index: 1\n", encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.fairino.ip, "10.0.0.7")
            self.assertEqual(cfg.fairino.gripper.index, 1)
            self.assertEqual(cfg.fairino.gripper.force_percent, Config().fairino.gripper.force_percent)
            with self.assertRaises(ConfigError):
                load_config(Path(tmp) / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
