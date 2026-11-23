"""Unit tests: joint set-point filter and operator-to-robot motion mapping."""

import math
import unittest

import numpy as np

from fr10.config import Config, MappingConfig
from fr10.mapping import PALM_DOWN, SHOULDER_HEIGHT, Workspace, WorkspaceMapper
from fr10.math_utils import exp_so3, make_pose, rot_x, rot_z, rotation_angle
from fr10.trajectory import JointMove, JointSetpointFilter, min_jerk

DT = 0.008


def make_filter(v_max=1.5, a_max=6.0, gain=15.0, n=3):
    return JointSetpointFilter([v_max] * n, [a_max] * n, [-3.0] * n, [3.0] * n, gain=gain)


def simulate(flt, target_fn, steps, dt=DT, scale=1.0):
    positions = [flt.position]
    for k in range(steps):
        positions.append(flt.step(target_fn(k), dt, scale))
    positions = np.array(positions)
    velocities = np.diff(positions, axis=0) / dt
    accelerations = np.diff(velocities, axis=0) / dt
    return positions, velocities, accelerations


class SetpointFilterTests(unittest.TestCase):
    def test_step_response_respects_limits_and_does_not_overshoot(self):
        for size in (0.002, 0.02, 0.2, 1.0, 2.5):
            with self.subTest(step=size):
                flt = make_filter()
                flt.reset(np.zeros(3))
                target = np.array([size, -size, 0.5 * size])
                pos, vel, acc = simulate(flt, lambda k: target, 2000)
                self.assertLessEqual(np.max(np.abs(vel)), 1.5 + 1e-9)
                self.assertLessEqual(np.max(np.abs(acc)), 6.0 + 1e-6)
                overshoot = np.max((pos - target) * np.sign(target), axis=0)
                self.assertLess(np.max(overshoot), max(1e-4, 0.002 * size))
                np.testing.assert_allclose(pos[-1], target, atol=1e-6)

    def test_settles_quickly_for_small_moves(self):
        flt = make_filter()
        flt.reset(np.zeros(3))
        pos, _, _ = simulate(flt, lambda k: np.full(3, 0.01), int(0.4 / DT))
        self.assertLess(np.max(np.abs(pos[-1] - 0.01)), 5e-4)  # within 5 % after 0.4 s

    def test_tracks_a_moving_target_with_bounded_lag(self):
        flt = make_filter()
        flt.reset(np.zeros(3))
        speed = 0.5  # rad/s, below the limit
        pos, vel, acc = simulate(flt, lambda k: np.full(3, speed * (k + 1) * DT), 500)
        lag = speed * 500 * DT - pos[-1]
        self.assertTrue(np.all(lag > 0.0) and np.all(lag < speed / 15.0 * 1.2))  # ~ v / gain
        np.testing.assert_allclose(vel[-1], speed, atol=1e-3)

    def test_noisy_target_is_smoothed(self):
        rng = np.random.default_rng(0)
        flt = make_filter()
        flt.reset(np.zeros(3))
        pos, vel, _ = simulate(flt, lambda k: rng.normal(0.0, 0.002, 3), 1000)
        self.assertLess(np.std(pos[200:]), 0.0012)  # less than the input noise

    def test_target_reversal_decelerates_within_limit(self):
        flt = make_filter()
        flt.reset(np.zeros(3))
        _, vel, acc = simulate(flt, lambda k: np.full(3, 2.0 if k < 60 else -2.0), 600)
        self.assertLessEqual(np.max(np.abs(acc)), 6.0 + 1e-6)
        self.assertLessEqual(np.max(np.abs(vel)), 1.5 + 1e-9)

    def test_speed_scale(self):
        flt = make_filter()
        flt.reset(np.zeros(3))
        _, vel, acc = simulate(flt, lambda k: np.full(3, 2.5), 400, scale=0.3)
        self.assertLessEqual(np.max(np.abs(vel)), 0.45 + 1e-9)
        self.assertLessEqual(np.max(np.abs(acc)), 1.8 + 1e-6)

    def test_position_limits(self):
        flt = make_filter()
        flt.reset(np.full(3, 2.9))
        pos, _, _ = simulate(flt, lambda k: np.full(3, 10.0), 500)
        self.assertLessEqual(np.max(pos), 3.0 + 1e-12)
        np.testing.assert_allclose(pos[-1], 3.0, atol=1e-6)

    def test_irregular_time_steps(self):
        rng = np.random.default_rng(1)
        flt = make_filter()
        flt.reset(np.zeros(3))
        target = np.full(3, 1.0)
        previous = flt.position
        for _ in range(3000):
            dt = float(rng.uniform(0.002, 0.03))
            current = flt.step(target, dt)
            self.assertLessEqual(np.max(np.abs(current - previous)) / dt, 1.5 + 1e-9)
            previous = current
        self.assertLess(np.max(previous - target), 2e-3)
        np.testing.assert_allclose(previous, target, atol=1e-4)

    def test_invalid_use(self):
        flt = make_filter()
        with self.assertRaises(RuntimeError):
            flt.step(np.zeros(3), DT)
        flt.reset(np.zeros(3))
        for bad in (np.array([0.0, np.nan, 0.0]), np.zeros(2)):
            with self.assertRaises(ValueError):
                flt.step(bad, DT)
        with self.assertRaises(ValueError):
            flt.step(np.zeros(3), 0.0)
        with self.assertRaises(ValueError):
            flt.reset(np.array([0.0, np.inf, 0.0]))
        with self.assertRaises(ValueError):
            JointSetpointFilter([1.0], [1.0], [1.0], [0.0])

    def test_velocity_scale_slows_without_weakening_the_brakes(self):
        flt = make_filter()
        flt.reset(np.full(3, -2.9))
        for _ in range(150):  # up to speed on a long trip (5.8 rad)
            flt.step(np.full(3, 2.9), DT)
        fast = float(np.max(flt.velocity))
        self.assertGreater(fast, 1.0)
        slowed = []  # the arm lags: slow down to a tenth of the speed limit
        for _ in range(100):
            flt.step(np.full(3, 2.9), DT, velocity_scale=0.1)
            slowed.append(float(np.max(flt.velocity)))
        self.assertLess(slowed[0], fast)
        # it decelerates at the full acceleration limit (6 rad/s^2) to the lower speed limit (0.15 rad/s)
        self.assertAlmostEqual(slowed[-1], 0.15, places=2)
        self.assertLessEqual(max(np.diff([fast] + slowed)) / DT, 1e-9 + 0.0)
        self.assertGreaterEqual(min(np.diff([fast] + slowed)) / DT, -6.0 - 1e-6)

    def test_joint_move_profile(self):
        self.assertEqual(min_jerk(0.0), 0.0)
        self.assertEqual(min_jerk(1.0), 1.0)
        self.assertAlmostEqual(min_jerk(0.5), 0.5)
        move = JointMove(np.zeros(3), np.array([1.0, -0.5, 0.0]), max_speed=0.5)
        self.assertAlmostEqual(move.duration, 1.875 * 1.0 / 0.5)
        points = [move.step(DT) for _ in range(int(move.duration / DT) + 5)]
        self.assertTrue(move.done)
        np.testing.assert_allclose(points[-1], [1.0, -0.5, 0.0], atol=1e-12)
        speed = np.abs(np.diff(np.array(points), axis=0)) / DT
        self.assertLessEqual(np.max(speed), 0.5 + 1e-3)
        self.assertEqual(JointMove(np.zeros(3), np.zeros(3), 0.5).duration, 1.0)  # minimum duration


def fast_mapping(**changes) -> MappingConfig:
    """Mapping without smoothing or speed limits, so outputs equal the raw mapping."""
    cfg = Config().mapping
    cfg.orientation_mode = "relative"
    cfg.smoothing_time_s = 0.0
    cfg.max_linear_speed = 1e6
    cfg.max_angular_speed_deg = 1e6
    cfg.jump_position_m = 1e6
    cfg.jump_rotation_deg = 179.0
    for key, value in changes.items():
        setattr(cfg, key, value)
    return cfg


def relative_mapping() -> MappingConfig:
    cfg = Config().mapping
    cfg.orientation_mode = "relative"
    return cfg


TCP0 = make_pose(rot_x(math.pi), [0.6, 0.0, 0.4])  # tool pointing down
HAND0 = make_pose(rot_z(0.3) @ rot_x(0.2), [0.2, -0.3, 1.1])
HEAD_FORWARD = make_pose(np.eye(3), [0.0, 0.0, 1.6])


class MappingTests(unittest.TestCase):
    def test_translation_follows_hand_one_to_one(self):
        mapper = WorkspaceMapper(fast_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, yaw_offset=0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.10, -0.05, 0.02]
        out = mapper.update(hand, 1 / 60)
        np.testing.assert_allclose(out.target[:3, 3], TCP0[:3, 3] + [0.10, -0.05, 0.02], atol=1e-12)
        np.testing.assert_allclose(out.target[:3, :3], TCP0[:3, :3], atol=1e-12)
        self.assertFalse(out.limited or out.jump)

    def test_no_motion_at_engage(self):
        mapper = WorkspaceMapper(relative_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, yaw_offset=0.4)
        out = mapper.update(HAND0, 1 / 60)
        np.testing.assert_allclose(out.target, TCP0, atol=1e-12)

    def test_position_scale(self):
        mapper = WorkspaceMapper(fast_mapping(position_scale=2.0))
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.05, 0.0, 0.0]
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, 3], TCP0[:3, 3] + [0.10, 0, 0], atol=1e-12)

    def test_directions_are_relative_to_head_yaw(self):
        """Operator faces +y of the headset world: 'forward' for them must be robot +x."""
        mapper = WorkspaceMapper(fast_mapping())
        head = make_pose(rot_z(math.pi / 2), [0, 0, 1.6])
        mapper.engage(HAND0, head, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.0, 0.10, 0.0]  # 10 cm along the operator's forward direction
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, 3], TCP0[:3, 3] + [0.10, 0, 0], atol=1e-12)

    def test_head_pitch_does_not_tilt_the_mapping(self):
        mapper = WorkspaceMapper(fast_mapping())
        head = make_pose(rot_z(0.5) @ np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]]), [0, 0, 1.6])  # looking down
        mapper.engage(HAND0, head, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.0, 0.0, 0.1]  # straight up
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, 3], TCP0[:3, 3] + [0, 0, 0.1], atol=1e-12)

    def test_world_reference_ignores_head(self):
        mapper = WorkspaceMapper(fast_mapping(reference="world"))
        mapper.engage(HAND0, make_pose(rot_z(1.0), [0, 0, 1.6]), TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.1, 0.0, 0.0]
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, 3], TCP0[:3, 3] + [0.1, 0, 0], atol=1e-12)

    def test_yaw_offset_rotates_motion(self):
        mapper = WorkspaceMapper(fast_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, yaw_offset=math.pi / 2)
        hand = HAND0.copy()
        hand[:3, 3] += [0.1, 0.0, 0.0]  # operator forward
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, 3], TCP0[:3, 3] + [0, 0.1, 0], atol=1e-12)

    def test_rotation_about_vertical_is_reproduced(self):
        mapper = WorkspaceMapper(fast_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, :3] = rot_z(0.4) @ HAND0[:3, :3]  # hand yaws left by 0.4 rad (world frame)
        out = mapper.update(hand, 0.01)
        np.testing.assert_allclose(out.target[:3, :3], rot_z(0.4) @ TCP0[:3, :3], atol=1e-12)
        np.testing.assert_allclose(out.target[:3, 3], TCP0[:3, 3], atol=1e-12)

    def test_rotation_is_expressed_in_operator_frame(self):
        """With a 90 deg yaw offset, rolling the hand about operator-forward rolls the tool about robot +y."""
        mapper = WorkspaceMapper(fast_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, yaw_offset=math.pi / 2)
        hand = HAND0.copy()
        hand[:3, :3] = rot_x(0.3) @ HAND0[:3, :3]
        expected = exp_so3([0.0, 0.3, 0.0]) @ TCP0[:3, :3]
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, :3], expected, atol=1e-12)

    def test_rotation_scale_and_orientation_lock(self):
        mapper = WorkspaceMapper(fast_mapping(rotation_scale=0.5))
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, :3] = rot_z(0.4) @ HAND0[:3, :3]
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, :3], rot_z(0.2) @ TCP0[:3, :3], atol=1e-9)
        locked = WorkspaceMapper(fast_mapping(track_orientation=False))
        locked.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        np.testing.assert_allclose(locked.update(hand, 0.01).target[:3, :3], TCP0[:3, :3], atol=1e-12)

    def test_reclutch_keeps_tool_where_it_is(self):
        mapper = WorkspaceMapper(fast_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.1, 0.0, 0.0]
        reached = mapper.update(hand, 0.01).target
        mapper.release()
        self.assertFalse(mapper.engaged)
        np.testing.assert_allclose(mapper.target, reached)
        elsewhere = make_pose(rot_z(-1.0), [0.9, 0.9, 0.9])  # hand moved freely while released
        mapper.engage(elsewhere, HEAD_FORWARD, reached, 0.0)
        np.testing.assert_allclose(mapper.update(elsewhere, 0.01).target, reached, atol=1e-12)
        with self.assertRaises(RuntimeError):
            released = WorkspaceMapper(fast_mapping())
            released.update(HAND0, 0.01)

    def test_workspace_limits(self):
        ws = Config().mapping.workspace
        mapper = WorkspaceMapper(fast_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.0, 0.0, -2.0]  # far below the table
        out = mapper.update(hand, 0.01)
        self.assertTrue(out.limited)
        self.assertAlmostEqual(out.target[2, 3], ws.z[0])
        workspace = Workspace(ws)
        rng = np.random.default_rng(0)
        for _ in range(5000):
            p = workspace.clamp(rng.uniform(-3, 3, 3))
            self.assertTrue(ws.x[0] - 1e-9 <= p[0] <= ws.x[1] + 1e-9)
            self.assertTrue(ws.y[0] - 1e-9 <= p[1] <= ws.y[1] + 1e-9)
            self.assertTrue(ws.z[0] - 1e-9 <= p[2] <= ws.z[1] + 1e-9)
            self.assertLessEqual(np.linalg.norm(p - [0, 0, SHOULDER_HEIGHT]), ws.max_reach + 1e-9)
            self.assertGreaterEqual(np.hypot(p[0], p[1]), ws.min_radius - 1e-9)
        inside = np.array([0.6, 0.1, 0.3])
        np.testing.assert_allclose(workspace.clamp(inside), inside)
        # clamping is the nearest point for the simple (single limit) cases
        np.testing.assert_allclose(workspace.clamp([0.6, 0.1, -1.0]), [0.6, 0.1, ws.z[0]])
        np.testing.assert_allclose(workspace.clamp([0.6, 2.0, 0.3]), [0.6, ws.y[1], 0.3])
        # a workspace that surrounds the base (box centre on the base axis) still works
        around = Config().mapping.workspace
        around.x = [-0.9, 0.9]
        ring = Workspace(around)
        self.assertGreaterEqual(np.hypot(*ring.clamp([0.0, 0.0, 0.5])[:2]), around.min_radius - 1e-9)
        empty = Config().mapping.workspace
        empty.x, empty.y, empty.min_radius = [-0.1, 0.1], [-0.1, 0.1], 0.3
        with self.assertRaises(ValueError):
            Workspace(empty)

    def test_speed_limits_on_target(self):
        cfg = Config().mapping
        cfg.orientation_mode = "relative"
        cfg.smoothing_time_s = 0.0
        cfg.jump_position_m = 10.0
        cfg.jump_rotation_deg = 179.0
        mapper = WorkspaceMapper(cfg)
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.3, 0.0, 0.0]
        hand[:3, :3] = rot_z(1.0) @ HAND0[:3, :3]
        dt = 1 / 60
        previous = TCP0
        for _ in range(200):
            out = mapper.update(hand, dt).target
            self.assertLessEqual(np.linalg.norm(out[:3, 3] - previous[:3, 3]), cfg.max_linear_speed * dt + 1e-9)
            self.assertLessEqual(
                rotation_angle(previous[:3, :3].T @ out[:3, :3]), math.radians(cfg.max_angular_speed_deg) * dt + 1e-9
            )
            previous = out
        np.testing.assert_allclose(previous[:3, 3], TCP0[:3, 3] + [0.3, 0, 0], atol=1e-9)
        np.testing.assert_allclose(previous[:3, :3], rot_z(1.0) @ TCP0[:3, :3], atol=1e-9)

    def test_smoothing_converges(self):
        mapper = WorkspaceMapper(relative_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.02, 0.0, 0.0]
        first = mapper.update(hand, 1 / 60).target
        self.assertTrue(0.0 < first[0, 3] - TCP0[0, 3] < 0.02)
        for _ in range(120):
            out = mapper.update(hand, 1 / 60).target
        np.testing.assert_allclose(out[:3, 3], TCP0[:3, 3] + [0.02, 0, 0], atol=1e-6)

    def test_tracking_jump_releases_clutch(self):
        mapper = WorkspaceMapper(relative_mapping())
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.5, 0.0, 0.0]  # teleport
        out = mapper.update(hand, 1 / 60)
        self.assertTrue(out.jump)
        self.assertFalse(mapper.engaged)
        np.testing.assert_allclose(out.target, TCP0, atol=1e-12)
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        self.assertTrue(mapper.update(np.zeros((4, 4)), 1 / 60).jump)  # invalid pose
        with self.assertRaises(ValueError):
            mapper.engage(np.zeros((4, 4)), HEAD_FORWARD, TCP0, 0.0)

    def test_missing_head_pose_falls_back(self):
        mapper = WorkspaceMapper(fast_mapping())
        mapper.engage(HAND0, None, TCP0, 0.0)
        hand = HAND0.copy()
        hand[:3, 3] += [0.1, 0.0, 0.0]
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, 3], TCP0[:3, 3] + [0.1, 0, 0], atol=1e-12)


# tool orientation of the default home pose: gripper pointing down, jaws opening along the base y axis
HOME_TOOL = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def absolute_mapping(**changes) -> MappingConfig:
    cfg = fast_mapping(orientation_mode="absolute", **changes)
    return cfg


def palm_down_hand(hand="right", yaw=0.0, position=(0.3, -0.2, 1.1)) -> np.ndarray:
    """Hand pose with the palm facing down and the fingers pointing 'forward' (rotated by ``yaw``)."""
    return make_pose(rot_z(yaw) @ PALM_DOWN[hand], position)


class SmoothingTests(unittest.TestCase):
    """Speed adaptive smoothing: strong at rest (kills tracking jitter), light during real motion."""

    def run_hand(self, hand_x, seconds=2.0, dt=1 / 60, **changes):
        cfg = relative_mapping()
        cfg.jump_position_m = 10.0
        for key, value in changes.items():
            setattr(cfg, key, value)
        mapper = WorkspaceMapper(cfg)
        mapper.engage(HAND0, HEAD_FORWARD, TCP0, 0.0)
        out = []
        for k in range(int(seconds / dt)):
            hand = HAND0.copy()
            hand[:3, 3] += [hand_x(k * dt), 0.0, 0.0]
            out.append(mapper.update(hand, dt).target[0, 3] - TCP0[0, 3])
        return np.array(out)

    def test_tracking_jitter_is_attenuated_at_rest(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.004, 120)  # 4 mm rms hand-tracking noise
        out = self.run_hand(lambda t: noise[int(round(t * 60)) % 120])
        self.assertLess(np.std(out[20:]), 0.5 * np.std(noise))
        old = self.run_hand(lambda t: noise[int(round(t * 60)) % 120], smoothing_time_s=0.04)
        self.assertLess(np.std(out[20:]), np.std(old[20:]))

    def test_deliberate_motion_is_not_delayed_much(self):
        out = self.run_hand(lambda t: 0.30 * t, seconds=1.5)  # constant 0.3 m/s
        lag = 0.30 * (len(out) / 60.0) - out[-1]
        self.assertLess(lag / 0.30, 0.09)  # under 90 ms behind a 0.3 m/s hand

    def test_converges_to_a_held_position(self):
        out = self.run_hand(lambda t: 0.05, seconds=2.0)
        self.assertAlmostEqual(out[-1], 0.05, delta=1e-5)

    def test_smoothing_can_be_switched_off(self):
        out = self.run_hand(lambda t: 0.05 * min(1.0, t * 10), seconds=0.5, smoothing_time_s=0.0)
        self.assertAlmostEqual(out[-1], 0.05, delta=1e-9)


class AbsoluteOrientationTests(unittest.TestCase):
    def mapper(self, hand="right", **changes):
        return WorkspaceMapper(absolute_mapping(**changes), hand=hand, neutral_rotation=HOME_TOOL)

    def test_palm_down_means_gripper_down_for_both_hands(self):
        for hand in ("right", "left"):
            with self.subTest(hand=hand):
                mapper = self.mapper(hand)
                tool = make_pose(HOME_TOOL, [0.6, 0.0, 0.4])
                hand_pose = palm_down_hand(hand)
                self.assertLess(mapper.alignment_error(hand_pose, HEAD_FORWARD, tool, 0.0), 1e-6)
                mapper.engage(hand_pose, HEAD_FORWARD, tool, 0.0)
                out = mapper.update(hand_pose, 0.01).target
                np.testing.assert_allclose(out[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)  # tool z points down
                np.testing.assert_allclose(out[:3, :3], HOME_TOOL, atol=1e-9)

    def test_palm_up_points_the_gripper_up(self):
        mapper = self.mapper("right")
        up = make_pose(rot_x(math.pi) @ PALM_DOWN["right"], [0.3, -0.2, 1.1])  # palm flipped to face up
        mapper.engage(up, HEAD_FORWARD, make_pose(HOME_TOOL, [0.6, 0, 0.4]), 0.0)
        out = mapper.update(up, 0.01).target
        self.assertGreater(out[2, 2], 0.99)  # tool z points up

    def test_engaged_posture_does_not_matter_after_engaging_palm_down(self):
        """Squeezing with the palm sideways engages only after turning the palm down."""
        mapper = self.mapper("right")
        tool = make_pose(HOME_TOOL, [0.6, 0, 0.4])
        sideways = make_pose(exp_so3([math.pi / 2, 0, 0]) @ PALM_DOWN["right"], [0.3, -0.2, 1.1])
        self.assertAlmostEqual(mapper.alignment_error(sideways, HEAD_FORWARD, tool, 0.0), 90.0, delta=0.01)

    def test_hand_turns_gripper_the_same_way(self):
        mapper = self.mapper("right")
        tool = make_pose(HOME_TOOL, [0.6, 0, 0.4])
        mapper.engage(palm_down_hand(), HEAD_FORWARD, tool, 0.0)
        # yaw the palm-down hand 40 deg to the left: the gripper turns 40 deg about the vertical
        out = mapper.update(palm_down_hand(yaw=math.radians(40)), 0.01).target
        np.testing.assert_allclose(out[:3, :3], rot_z(math.radians(40)) @ HOME_TOOL, atol=1e-9)
        # roll the hand about its finger axis (forward): the gripper tilts about the forward axis
        rolled = make_pose(palm_down_hand()[:3, :3] @ rot_x(math.radians(20)), [0.3, -0.2, 1.1])
        out = mapper.update(rolled, 0.01).target
        np.testing.assert_allclose(out[:3, :3], exp_so3([0.0, 0.0, 0.0]) @ (rolled[:3, :3] @ PALM_DOWN["right"].T @ HOME_TOOL), atol=1e-9)
        self.assertAlmostEqual(math.degrees(rotation_angle(HOME_TOOL.T @ out[:3, :3])), 20.0, delta=0.01)

    def test_directions_follow_the_operator_facing_direction(self):
        """Operator faces +y of the headset world and holds the palm down: still 'gripper down, jaws sideways'."""
        mapper = self.mapper("right")
        head = make_pose(rot_z(math.pi / 2), [0, 0, 1.6])
        tool = make_pose(HOME_TOOL, [0.6, 0, 0.4])
        hand = palm_down_hand(yaw=math.pi / 2)  # fingers pointing where the operator faces
        self.assertLess(mapper.alignment_error(hand, head, tool, 0.0), 1e-6)
        mapper.engage(hand, head, tool, 0.0)
        np.testing.assert_allclose(mapper.update(hand, 0.01).target[:3, :3], HOME_TOOL, atol=1e-9)

    def test_rotation_scale_and_orientation_lock(self):
        tool = make_pose(HOME_TOOL, [0.6, 0, 0.4])
        mapper = self.mapper("right", rotation_scale=0.5)
        mapper.engage(palm_down_hand(), HEAD_FORWARD, tool, 0.0)
        out = mapper.update(palm_down_hand(yaw=math.radians(40)), 0.01).target
        np.testing.assert_allclose(out[:3, :3], rot_z(math.radians(20)) @ HOME_TOOL, atol=1e-9)
        locked = self.mapper("right", track_orientation=False)
        self.assertEqual(locked.alignment_error(palm_down_hand(yaw=1.0), HEAD_FORWARD, tool, 0.0), 0.0)
        locked.engage(palm_down_hand(), HEAD_FORWARD, tool, 0.0)
        np.testing.assert_allclose(locked.update(palm_down_hand(yaw=1.0), 0.01).target[:3, :3], HOME_TOOL, atol=1e-9)

    def test_hand_offset_retunes_the_palm_down_reference(self):
        tool = make_pose(HOME_TOOL, [0.6, 0, 0.4])
        tilted = make_pose(palm_down_hand()[:3, :3] @ rot_x(math.radians(30)), [0.3, -0.2, 1.1])
        plain = self.mapper("right")
        self.assertAlmostEqual(plain.alignment_error(tilted, HEAD_FORWARD, tool, 0.0), 30.0, delta=0.01)
        tuned = self.mapper("right", hand_offset_rpy_deg=[-30.0, 0.0, 0.0])
        self.assertLess(tuned.alignment_error(tilted, HEAD_FORWARD, tool, 0.0), 1e-6)

    def test_relative_mode_ignores_alignment(self):
        tool = make_pose(HOME_TOOL, [0.6, 0, 0.4])
        mapper = WorkspaceMapper(fast_mapping(), hand="right", neutral_rotation=HOME_TOOL)
        self.assertEqual(mapper.alignment_error(make_pose(rot_x(2.0), [0, 0, 1]), HEAD_FORWARD, tool, 0.0), 0.0)

    def test_invalid_hand_name(self):
        with self.assertRaises(ValueError):
            WorkspaceMapper(Config().mapping, hand="both")


if __name__ == "__main__":
    unittest.main()
