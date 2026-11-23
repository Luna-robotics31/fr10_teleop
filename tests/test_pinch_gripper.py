"""Standalone pinch -> gripper tool: detection, the gripper.py calls, and the open/close logic."""

import unittest

import numpy as np

from fr10.config import Config
from fr10.pinch_gripper import (
    PinchDetector, PinchGripperController, ScriptGripper, pinch_distance,
)
from fr10.robots.base import RobotError


def hand(gap=None):
    """25 joint positions with the thumb tip ``gap`` metres from the index fingertip."""
    points = np.zeros((25, 3))
    points[:, 0] = 0.05  # all joints somewhere in front of the origin (a tracked hand)
    points[9] = [0.2, 0.0, 0.0]
    points[4] = [0.2, 0.0 if gap is None else gap, 0.0]
    return points


class FakeRPC:
    def __init__(self, act=0, move=0):
        self.calls, self.act, self.move = [], act, move

    def ActGripper(self, index, action):
        self.calls.append(("ActGripper", index, action))
        return self.act

    def MoveGripper(self, *args):
        self.calls.append(("MoveGripper",) + args)
        return self.move


class PinchDistanceTests(unittest.TestCase):
    def test_distance_and_untracked_hands(self):
        self.assertAlmostEqual(pinch_distance(hand(0.012)), 0.012)
        self.assertIsNone(pinch_distance(np.zeros((25, 3))))  # an untracked hand sits at the origin
        self.assertIsNone(pinch_distance(np.zeros((24, 3))))
        bad = hand(0.01)
        bad[4, 1] = np.nan
        self.assertIsNone(pinch_distance(bad))


class PinchDetectorTests(unittest.TestCase):
    def test_hold_time_hysteresis_and_release_delay(self):
        det = PinchDetector(on_m=0.03, off_m=0.05, hold_s=0.08, release_s=0.10)
        self.assertFalse(det.update(0.09, 0.0))
        self.assertFalse(det.update(0.02, 0.01))  # just started
        self.assertFalse(det.update(0.02, 0.05))  # not held long enough
        self.assertTrue(det.update(0.02, 0.10))  # held 0.09 s
        for t, d in ((0.12, 0.04), (0.14, 0.049), (0.16, 0.03)):  # jitter between the thresholds
            self.assertTrue(det.update(d, t))
        self.assertTrue(det.update(0.08, 0.20))  # open, but only just
        self.assertTrue(det.update(0.08, 0.28))
        self.assertFalse(det.update(0.08, 0.31))  # open for more than 0.10 s

    def test_glitches_do_nothing(self):
        det = PinchDetector()
        self.assertFalse(det.update(0.02, 0.0))
        self.assertFalse(det.update(0.09, 0.03))  # a single frame is not a pinch
        self.assertFalse(det.update(0.02, 0.04))
        self.assertFalse(det.update(0.02, 0.10))  # restarted at 0.04 s: only 0.06 s so far
        self.assertTrue(det.update(0.02, 0.13))
        self.assertTrue(det.update(None, 0.14))  # one lost frame while pinching
        self.assertTrue(det.update(0.02, 0.15))

    def test_lost_hand_releases_after_the_delay_and_never_starts_a_pinch(self):
        det = PinchDetector()
        for t in (0.0, 0.2, 0.4):
            self.assertFalse(det.update(None, t))
        det.update(0.02, 0.5)
        det.update(0.02, 0.7)
        self.assertTrue(det.pinched)
        det.update(None, 0.8)
        self.assertFalse(det.update(None, 1.0))

    def test_invalid_settings(self):
        for kwargs in ({"on_m": 0.05, "off_m": 0.03}, {"on_m": 0.0}, {"hold_s": -1.0}):
            with self.assertRaises(ValueError):
                PinchDetector(**kwargs)


class ScriptGripperTests(unittest.TestCase):
    def test_open_and_close_are_the_calls_of_gripper_py(self):
        rpc = FakeRPC()
        gripper = ScriptGripper(Config().validate(), rpc)
        gripper.open_gripper()
        gripper.close_gripper()
        self.assertEqual(
            rpc.calls,
            [
                ("ActGripper", 2, 1),
                ("MoveGripper", 2, 0, 100, 67, 2000, 0, 0, 0, 0, 0),
                ("ActGripper", 2, 1),
                ("MoveGripper", 2, 100, 100, 67, 2000, 0, 0, 0, 0, 0),
            ],
        )

    def test_one_activation_then_fast_moves(self):
        """A program that follows a hand activates once and then sends moves that return at once."""
        rpc = FakeRPC()
        gripper = ScriptGripper(Config().validate(), rpc)
        gripper.activate()
        gripper.activate_each_move = False
        gripper.block = 1
        gripper.move_percent(40)
        gripper.close_gripper()
        gripper.open_gripper()
        self.assertEqual(
            rpc.calls,
            [
                ("ActGripper", 2, 1),
                ("MoveGripper", 2, 40, 100, 67, 2000, 1, 0, 0, 0, 0),
                ("MoveGripper", 2, 100, 100, 67, 2000, 1, 0, 0, 0, 0),
                ("MoveGripper", 2, 0, 100, 67, 2000, 1, 0, 0, 0, 0),
            ],
        )

    def test_activation_failure_and_defaults_stay_like_gripper_py(self):
        with self.assertRaisesRegex(RobotError, "ActGripper.*error code 3"):
            ScriptGripper(Config().validate(), FakeRPC(act=3)).activate()
        rpc = FakeRPC()
        ScriptGripper(Config().validate(), rpc).move_percent(10)  # defaults: activate first, wait for the move
        self.assertEqual([c[0] for c in rpc.calls], ["ActGripper", "MoveGripper"])
        self.assertEqual(rpc.calls[1][6], 0)

    def test_values_come_from_the_configuration(self):
        cfg = Config()
        cfg.fairino.gripper.index = 1
        cfg.fairino.gripper.speed_percent = 40
        cfg.fairino.gripper.force_percent = 20
        cfg.fairino.gripper.max_time_ms = 1500
        cfg.fairino.gripper.closed_percent = 80.0
        rpc = FakeRPC()
        ScriptGripper(cfg.validate(), rpc).close_gripper()
        self.assertEqual(rpc.calls[1], ("MoveGripper", 1, 80, 40, 20, 1500, 0, 0, 0, 0, 0))

    def test_refusals_raise_with_the_error_code(self):
        with self.assertRaisesRegex(RobotError, "ActGripper.*error code 3"):
            ScriptGripper(Config().validate(), FakeRPC(act=3)).close_gripper()
        rpc = FakeRPC(move=9)
        with self.assertRaisesRegex(RobotError, "MoveGripper.*error code 9"):
            ScriptGripper(Config().validate(), rpc).open_gripper()
        with self.assertRaisesRegex(RobotError, "not connected"):
            ScriptGripper(Config().validate(), None).open_gripper()


class RecordingGripper:
    def __init__(self):
        self.calls, self.fail = [], 0

    def _do(self, name):
        if self.fail:
            self.fail -= 1
            raise RobotError("refused")
        self.calls.append(name)

    def open_gripper(self):
        self._do("OPEN")

    def close_gripper(self):
        self._do("CLOSE")


def run(controller, samples, start=0.0, dt=0.02):
    """Feed (distance) samples at 50 Hz; returns the commands accepted."""
    done = []
    for i, distance in enumerate(samples):
        action = controller.update(distance, start + i * dt)
        if action:
            done.append(action)
    return done


class ControllerTests(unittest.TestCase):
    def test_pinch_closes_and_release_opens(self):
        gripper = RecordingGripper()
        controller = PinchGripperController(gripper)
        self.assertEqual(run(controller, [0.09] * 20), [])  # nothing at start
        self.assertEqual(run(controller, [0.02] * 20, start=1.0), ["CLOSE"])
        self.assertEqual(run(controller, [0.02] * 50, start=2.0), [])  # holding: no repeats
        self.assertEqual(run(controller, [0.09] * 20, start=4.0), ["OPEN"])
        self.assertEqual(gripper.calls, ["CLOSE", "OPEN"])
        self.assertEqual(controller.state, "open")

    def test_noise_around_the_threshold_sends_nothing(self):
        gripper = RecordingGripper()
        controller = PinchGripperController(gripper)
        rng = np.random.default_rng(0)
        run(controller, list(0.02 + rng.uniform(-0.004, 0.004, 40)), start=0.0)  # a firm pinch
        run(controller, list(0.04 + rng.uniform(-0.009, 0.009, 200)), start=1.0)  # wobble inside 30..50 mm
        self.assertEqual(gripper.calls, ["CLOSE"])

    def test_refused_command_is_retried(self):
        gripper = RecordingGripper()
        gripper.fail = 2
        controller = PinchGripperController(gripper, retry_s=0.3)
        actions = run(controller, [0.02] * 100)  # 2 s
        self.assertEqual(actions, ["CLOSE"])
        self.assertEqual(controller.failures, 0)
        self.assertEqual([entry[2] for entry in controller.log], [False, False, True])

    def test_toggle_mode(self):
        gripper = RecordingGripper()
        controller = PinchGripperController(gripper, toggle=True)
        for k in range(3):
            run(controller, [0.02] * 15, start=k * 2.0)
            run(controller, [0.09] * 15, start=k * 2.0 + 1.0)
        self.assertEqual(gripper.calls, ["CLOSE", "OPEN", "CLOSE"])

    def test_open_first(self):
        gripper = RecordingGripper()
        controller = PinchGripperController(gripper, open_first=True)
        self.assertEqual(run(controller, [0.09] * 5), ["OPEN"])

    def test_lost_hand_while_pinching_keeps_the_gripper_closed_for_a_moment_only(self):
        gripper = RecordingGripper()
        controller = PinchGripperController(gripper)
        run(controller, [0.02] * 20, start=0.0)
        run(controller, [None] * 3, start=1.0)  # a short dropout
        self.assertEqual(gripper.calls, ["CLOSE"])
        run(controller, [None] * 30, start=2.0)  # the hand is really gone
        self.assertEqual(gripper.calls, ["CLOSE", "OPEN"])


if __name__ == "__main__":
    unittest.main()
