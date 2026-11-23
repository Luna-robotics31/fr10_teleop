"""Event freshness must depend on XR receipts, including stationary poses."""

import asyncio
import copy
from multiprocessing import Process
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fr10.config import Config
from fr10.xr_input import TeleVuerInput, fist_ratio


class FakeTeleVuer:
    """TeleVuer callback/registration contract, without a web server or headset."""

    def __init__(self, use_hand_tracking, **kwargs):
        self.ready_stamp_at_init = self.fr10_motion_time
        self.registered = self.on_hand_move if use_hand_tracking else self.on_controller_move
        self.motion_data_ready = False
        self.head_pose = np.eye(4)
        self.closed = False
        self.callbacks = 0
        for side in ("left", "right"):
            setattr(self, f"{side}_arm_pose", np.zeros((4, 4)))
            setattr(self, f"{side}_hand_positions", np.zeros((25, 3)))
            for suffix in ("ctrl_squeezeValue", "ctrl_triggerValue", "hand_pinchValue"):
                setattr(self, f"{side}_{suffix}", 0.0)
            for suffix in ("ctrl_aButton", "ctrl_bButton", "hand_pinch"):
                setattr(self, f"{side}_{suffix}", False)

    async def on_controller_move(self, event, _session, fps=60):
        self.callbacks += 1
        for side in ("left", "right"):
            setattr(self, f"{side}_arm_pose", np.array(event.value[side]).reshape(4, 4, order="F"))
            state = event.value[f"{side}State"]
            for suffix in ("squeezeValue", "triggerValue", "aButton", "bButton"):
                setattr(self, f"{side}_ctrl_{suffix}", state.get(suffix, 0.0))
        self.motion_data_ready = True

    async def on_hand_move(self, event, _session, fps=60):
        self.callbacks += 1
        for side in ("left", "right"):
            setattr(self, f"{side}_arm_pose", np.array(event.value[side][:16]).reshape(4, 4, order="F"))
            flat = np.array(event.value[side], dtype=float)
            setattr(self, f"{side}_hand_positions",
                    np.array([flat[i * 16 + 12: i * 16 + 15] for i in range(25)]))
            state = event.value[f"{side}State"]
            setattr(self, f"{side}_hand_pinch", bool(state.get("pinch", False)))
            setattr(self, f"{side}_hand_pinchValue", state.get("pinchValue", 0.0))
            setattr(self, f"{side}_hand_squeezeValue", state.get("squeezeValue", 0.0))
        self.motion_data_ready = True

    def close(self):
        self.closed = True


def make_event(hand_mode=False):
    flat = np.eye(4).reshape(16, order="F").tolist()
    raw = flat * (25 if hand_mode else 1)
    state = {"pinchValue": 0.04, "pinch": True} if hand_mode else {"squeezeValue": 0.8, "triggerValue": 0.4}
    return SimpleNamespace(value={"left": raw.copy(), "right": raw.copy(), "leftState": state.copy(), "rightState": state.copy()})


def set_shared_stamp(shared_stamp):
    with shared_stamp.get_lock():
        shared_stamp.value = 123.0


class XRFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.clock = patch("fr10.xr_input.time", SimpleNamespace(monotonic=lambda: self.now))
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.module = patch.dict("sys.modules", {"televuer": SimpleNamespace(TeleVuer=FakeTeleVuer)})
        self.module.start()
        self.addCleanup(self.module.stop)

    def source(self, hand_mode=False, side="right", **xr_options):
        cfg = Config()
        cfg.xr.input_mode = "hand" if hand_mode else "controller"
        cfg.xr.hand = side
        for key, value in xr_options.items():
            setattr(cfg.xr, key, value)
        source = TeleVuerInput(cfg, "pass-through")
        self.addCleanup(source.close)
        self.assertEqual(source._tv.ready_stamp_at_init, 0.0)
        return source

    def emit(self, source, event, method=None):
        callback = source._tv.registered if method is None else getattr(source._tv, method)
        asyncio.run(callback(event, None))

    def test_stationary_controller_receipts_remain_fresh_and_event_loss_expires(self):
        source = self.source()
        self.assertFalse(source.read().connected)
        event = make_event()
        for _ in range(10):
            self.emit(source, event)
            out = source.read()
            self.assertTrue(out.connected)
            self.assertTrue(out.valid)
            self.assertAlmostEqual(out.clutch, 0.8)
            self.now += 0.2  # Same pose for 2 seconds; much longer than stale_timeout_s.
        self.now += 0.5
        self.assertFalse(source.read().valid)
        self.emit(source, event)
        self.assertTrue(source.read().valid)

    def test_stationary_hand_receipts_and_left_selection_remain_fresh(self):
        source = self.source(hand_mode=True, side="left")
        event = make_event(hand_mode=True)
        for _ in range(6):
            self.emit(source, event)
            self.assertTrue(source.read().valid)
            self.now += 0.3
        self.now += 0.5
        self.assertFalse(source.read().valid)

    def test_malformed_packets_do_not_refresh_or_replace_last_valid_pose(self):
        for hand_mode in (False, True):
            source = self.source(hand_mode)
            good = make_event(hand_mode)
            self.emit(source, good)
            stamp = source._tv.fr10_motion_time
            held_pose = source.read().hand_pose.copy()
            self.now += 1.0
            bad_events = []
            for key, value in (
                ("right", [0.0] * (400 if hand_mode else 16)),
                ("right", [1.0, 2.0]), ("leftState", None),
                ("rightState", {"squeezeValue": float("nan")}),
            ):
                bad = copy.deepcopy(good)
                bad.value[key] = value
                bad_events.append(bad)
            missing = copy.deepcopy(good)
            del missing.value["right"]
            bad_events.append(missing)
            bad_events.append(SimpleNamespace(value={"camera": {"matrix": np.eye(4).reshape(16).tolist()}}))
            for event in bad_events:
                with self.subTest(hand_mode=hand_mode, event=event):
                    self.emit(source, event)
                    self.assertEqual(source._tv.fr10_motion_time, stamp)
                    self.assertFalse(source.read().valid)
                    np.testing.assert_array_equal(source.read().hand_pose, held_pose)

    def test_other_input_mode_does_not_refresh_and_opposite_side_cannot_mask_loss(self):
        source = self.source()
        self.emit(source, make_event())
        stamp = source._tv.fr10_motion_time
        self.now += 1.0
        self.emit(source, make_event(True), method="on_hand_move")
        self.assertEqual(source._tv.fr10_motion_time, stamp)
        event = make_event()
        event.value["right"] = [0.0] * 16  # Left still valid, selected right tracking lost.
        self.emit(source, event)
        self.assertFalse(source.read().valid)
        self.assertEqual(source._tv.callbacks, 1)

    def test_zero_opposite_pose_is_allowed_when_selected_pose_is_valid(self):
        source = self.source()
        event = make_event()
        event.value["left"] = [0.0] * 16
        self.emit(source, event)
        self.assertTrue(source.read().valid)

    def test_receipt_value_is_shared_with_child_process_and_upstream_close_is_used(self):
        source = self.source()
        process = Process(target=set_shared_stamp, args=(source._tv._fr10_motion_stamp,))
        process.start()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail("shared timestamp test process did not finish")
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(source._tv.fr10_motion_time, 123.0)
        source.close()
        self.assertTrue(source._tv.closed)


def hand_flat(ratio=None, tracked=True, finger_ratios=None, thumb_gap=None):
    """Flat 25 x 16 hand data.

    ``ratio``: fingertip/knuckle wrist distance of every finger (about 2 open, 1 fist);
    ``finger_ratios``: the same per finger (index, middle, ring, little);
    ``thumb_gap``: distance of the thumb tip from the index fingertip [m] (default: at the wrist).
    """
    joints = np.tile(np.eye(4), (25, 1, 1))
    if tracked:
        joints[:, :3, 3] = [0.0, 0.0, 0.0]
        if finger_ratios is None and ratio is not None:
            finger_ratios = [ratio] * 4
        if finger_ratios is not None:
            for index, r in zip((6, 11, 16, 21), (1.0, 1.0, 1.0, 1.0)):  # knuckles 9 cm from the wrist
                joints[index, :3, 3] = [0.09, 0.0, 0.0]
            for index, r in zip((9, 14, 19, 24), finger_ratios):  # fingertips
                joints[index, :3, 3] = [0.09 * r, 0.0, 0.0]
            if thumb_gap is not None:
                joints[4, :3, 3] = joints[9, :3, 3] + [0.0, thumb_gap, 0.0]
    return np.concatenate([m.reshape(16, order="F") for m in joints]).tolist() if tracked else [0.0] * 400


class FistRatioTests(unittest.TestCase):
    def positions(self, ratio):
        flat = np.array(hand_flat(ratio)).reshape(25, 16)
        return flat[:, 12:15]

    def test_open_hand_and_fist(self):
        self.assertAlmostEqual(fist_ratio(self.positions(2.0)), 2.0)
        self.assertAlmostEqual(fist_ratio(self.positions(1.0)), 1.0)

    def test_untracked_or_invalid_hands_have_no_ratio(self):
        self.assertIsNone(fist_ratio(np.zeros((25, 3))))  # untracked hands report the origin
        self.assertIsNone(fist_ratio(np.zeros((24, 3))))
        bad = self.positions(2.0).copy()
        bad[9, 0] = np.nan
        self.assertIsNone(fist_ratio(bad))

    def test_scale_independent(self):
        self.assertAlmostEqual(fist_ratio(1.3 * self.positions(1.7)), 1.7)


class HandModeInputTests(unittest.TestCase):
    """Hand mode: the clutch is a closed fist of the non-driving (left) hand."""

    setUp = XRFreshnessTests.setUp
    source = XRFreshnessTests.source
    emit = XRFreshnessTests.emit

    def feed(self, source, left_ratio, left_tracked=True, dt=0.0):
        self.now += dt
        event = make_event(hand_mode=True)
        event.value["left"] = hand_flat(left_ratio, left_tracked)
        event.value["right"] = hand_flat(2.0)
        self.emit(source, event)
        return source.read()

    def test_fist_closes_the_clutch_after_the_engage_delay_and_open_hand_releases(self):
        source = self.source(hand_mode=True)
        self.assertEqual(self.feed(source, 2.0).clutch, 0.0)  # open hand
        self.assertEqual(self.feed(source, 1.0, dt=0.01).clutch, 0.0)  # fist seen, not held long enough
        self.assertEqual(self.feed(source, 1.0, dt=0.05).clutch, 0.0)
        out = self.feed(source, 1.0, dt=0.06)  # held for 0.11 s
        self.assertEqual(out.clutch, 1.0)
        self.assertIn("fist ratio 1.00", out.detail)
        self.assertEqual(self.feed(source, 2.0, dt=0.01).clutch, 1.0)  # open for a moment: still held
        self.assertEqual(self.feed(source, 2.0, dt=0.15).clutch, 0.0)  # open longer than the release delay

    def test_hysteresis_between_the_thresholds(self):
        source = self.source(hand_mode=True)
        self.feed(source, 1.0)
        self.assertEqual(self.feed(source, 1.0, dt=0.2).clutch, 1.0)
        for ratio in (1.3, 1.4, 1.45, 1.26):  # half closed: keeps the clutch
            self.assertEqual(self.feed(source, ratio, dt=0.1).clutch, 1.0, ratio)
        self.feed(source, 2.0)
        self.assertEqual(self.feed(source, 2.0, dt=0.2).clutch, 0.0)  # opened long enough: released
        self.assertEqual(self.feed(source, 1.4, dt=0.3).clutch, 0.0)  # a half closed hand does not engage
        self.assertEqual(self.feed(source, 1.4, dt=0.3).clutch, 0.0)

    def test_a_brief_fist_or_dropout_does_nothing(self):
        source = self.source(hand_mode=True)
        self.feed(source, 2.0)
        self.assertEqual(self.feed(source, 1.0, dt=0.03).clutch, 0.0)
        self.assertEqual(self.feed(source, 2.0, dt=0.03).clutch, 0.0)
        self.assertEqual(self.feed(source, 1.0, dt=0.03).clutch, 0.0)
        self.assertEqual(self.feed(source, 1.0, dt=0.06).clutch, 0.0)  # restarted: only 0.06 s
        self.assertEqual(self.feed(source, 1.0, dt=0.06).clutch, 1.0)
        self.assertEqual(self.feed(source, 0.0, left_tracked=False, dt=0.02).clutch, 1.0)  # one lost frame
        self.assertEqual(self.feed(source, 1.0, dt=0.02).clutch, 1.0)

    def test_untracked_left_hand_never_engages_and_sustained_loss_releases(self):
        source = self.source(hand_mode=True)
        out = self.feed(source, None, left_tracked=False)
        self.assertEqual(out.clutch, 0.0)
        self.assertIn("NOT tracked", out.detail)
        self.assertEqual(self.feed(source, None, left_tracked=False, dt=1.0).clutch, 0.0)
        self.feed(source, 1.0)
        self.assertEqual(self.feed(source, 1.0, dt=0.2).clutch, 1.0)
        self.assertEqual(self.feed(source, None, left_tracked=False, dt=0.05).clutch, 1.0)
        self.assertEqual(self.feed(source, None, left_tracked=False, dt=0.2).clutch, 0.0)

    def test_driving_hand_fist_is_not_the_clutch(self):
        source = self.source(hand_mode=True)
        event = make_event(hand_mode=True)
        event.value["left"] = hand_flat(2.0)
        event.value["right"] = hand_flat(1.0)  # right hand closed
        for _ in range(3):
            self.now += 0.2
            self.emit(source, event)
        self.assertEqual(source.read().clutch, 0.0)

    def test_gripper_can_come_from_the_driving_hand(self):
        source = self.source(hand_mode=True, gripper_source="driving_hand_pinch")
        event = make_event(hand_mode=True)
        event.value["rightState"] = {"pinchValue": 0.02, "pinch": True}
        self.emit(source, event)
        self.assertAlmostEqual(source.read().grip, 1.0)
        event.value["rightState"] = {"pinchValue": 0.09, "pinch": False}
        self.emit(source, event)
        self.assertAlmostEqual(source.read().grip, 0.0)


class PinchGripperTests(unittest.TestCase):
    """Hand mode gripper option: a pinch (other fingers straight) of either hand toggles the gripper."""

    setUp = XRFreshnessTests.setUp
    emit = XRFreshnessTests.emit

    def source(self, hand_mode=True, side="right", **options):
        options.setdefault("gripper_source", "both_hands_pinch")
        return XRFreshnessTests.source(self, hand_mode, side, **options)

    def feed(self, source, left, dt=0.0, tracked=True, right=None):
        self.now += dt
        event = make_event(hand_mode=True)
        event.value["left"] = left if tracked else [0.0] * 400
        event.value["right"] = hand_flat(2.0) if right is None else right
        self.emit(source, event)
        return source.read()

    OPEN = dict(finger_ratios=[2.0, 2.0, 2.0, 2.0], thumb_gap=0.09)
    PINCH = dict(finger_ratios=[1.6, 2.0, 2.0, 2.0], thumb_gap=0.012)  # thumb on index, others straight
    FIST = dict(finger_ratios=[1.0, 1.0, 1.0, 1.0], thumb_gap=0.01)  # thumb also touches the index in a fist

    def test_pinch_with_straight_fingers_is_a_grip_signal_after_the_hold_time(self):
        source = self.source(hand_mode=True)
        self.assertEqual(self.feed(source, hand_flat(**self.OPEN)).grip, 0.0)
        self.assertEqual(self.feed(source, hand_flat(**self.PINCH), dt=0.02).grip, 0.0)  # not held long enough
        out = self.feed(source, hand_flat(**self.PINCH), dt=0.08)
        self.assertEqual(out.grip, 1.0)
        self.assertEqual(out.clutch, 0.0)  # a pinch is not a fist
        self.assertIn("pinch", out.detail)
        self.assertEqual(self.feed(source, hand_flat(**self.OPEN), dt=0.02).grip, 0.0)

    def test_a_fist_is_never_a_pinch(self):
        source = self.source(hand_mode=True)
        for _ in range(10):
            out = self.feed(source, hand_flat(**self.FIST), dt=0.05)
            self.assertEqual(out.grip, 0.0)
        self.assertEqual(out.clutch, 1.0)  # and it is the clutch

    def test_pinch_is_ignored_while_the_clutch_is_closed(self):
        source = self.source(hand_mode=True)
        for _ in range(5):
            self.feed(source, hand_flat(**self.FIST), dt=0.05)
        self.assertEqual(self.feed(source, hand_flat(**self.PINCH), dt=0.03).grip, 0.0)  # clutch still closing

    def test_pinch_hysteresis_and_glitches(self):
        source = self.source(hand_mode=True)
        self.feed(source, hand_flat(**self.PINCH))
        self.assertEqual(self.feed(source, hand_flat(**self.PINCH), dt=0.1).grip, 1.0)
        wobble = dict(self.PINCH, thumb_gap=0.04)  # between the on (30 mm) and off (50 mm) thresholds
        self.assertEqual(self.feed(source, hand_flat(**wobble), dt=0.05).grip, 1.0)
        self.assertEqual(self.feed(source, hand_flat(**self.OPEN), dt=0.05).grip, 0.0)
        glitch = self.feed(source, hand_flat(**self.PINCH), dt=0.02)  # a single frame is not a pinch
        self.assertEqual(glitch.grip, 0.0)
        self.assertEqual(self.feed(source, hand_flat(**self.OPEN), dt=0.02).grip, 0.0)

    def test_untracked_left_hand_gives_no_grip(self):
        source = self.source(hand_mode=True)
        for _ in range(5):
            self.assertEqual(self.feed(source, None, dt=0.1, tracked=False).grip, 0.0)

    def test_right_hand_pinch_toggles_the_gripper_too(self):
        source = self.source(hand_mode=True)
        self.feed(source, hand_flat(**self.OPEN), right=hand_flat(**self.OPEN))
        self.assertEqual(self.feed(source, hand_flat(**self.OPEN), dt=0.02, right=hand_flat(**self.PINCH)).grip, 0.0)
        out = self.feed(source, hand_flat(**self.OPEN), dt=0.08, right=hand_flat(**self.PINCH))
        self.assertEqual(out.grip, 1.0)
        self.assertIn("right pinch", out.detail)
        self.assertEqual(self.feed(source, hand_flat(**self.OPEN), dt=0.05, right=hand_flat(**self.OPEN)).grip, 0.0)

    def test_right_pinch_works_while_the_left_fist_holds_the_clutch(self):
        source = self.source(hand_mode=True)
        for _ in range(5):  # left fist: following
            out = self.feed(source, hand_flat(**self.FIST), dt=0.05, right=hand_flat(**self.OPEN))
        self.assertEqual(out.clutch, 1.0)
        self.feed(source, hand_flat(**self.FIST), dt=0.02, right=hand_flat(**self.PINCH))
        out = self.feed(source, hand_flat(**self.FIST), dt=0.08, right=hand_flat(**self.PINCH))
        self.assertEqual(out.clutch, 1.0)
        self.assertEqual(out.grip, 1.0)  # the gripper can be toggled without letting go of the clutch

    def test_right_fist_or_curled_right_hand_is_not_a_pinch(self):
        source = self.source(hand_mode=True)
        for _ in range(5):
            out = self.feed(source, hand_flat(**self.OPEN), dt=0.05, right=hand_flat(**self.FIST))
        self.assertEqual(out.grip, 0.0)
        self.assertEqual(out.clutch, 0.0)  # only the left hand is the clutch

    def test_other_hand_only_option_ignores_the_right_hand(self):
        source = self.source(hand_mode=True, gripper_source="other_hand_pinch")
        for _ in range(4):
            out = self.feed(source, hand_flat(**self.OPEN), dt=0.1, right=hand_flat(**self.PINCH))
        self.assertEqual(out.grip, 0.0)

    def test_right_hand_pinch_value_of_the_headset_is_ignored(self):
        source = self.source(hand_mode=True)
        event = make_event(hand_mode=True)
        event.value["left"] = hand_flat(**self.OPEN)
        event.value["right"] = hand_flat(2.0)
        event.value["rightState"] = {"pinchValue": 0.01, "pinch": True}
        for _ in range(4):
            self.now += 0.1
            self.emit(source, event)
        self.assertEqual(source.read().grip, 0.0)


class FistGripperTests(unittest.TestCase):
    """Default hand mode gripper: the closure of the right hand sets the gripper opening."""

    setUp = XRFreshnessTests.setUp
    source = XRFreshnessTests.source
    emit = XRFreshnessTests.emit

    def settle(self, source, right, left=None, steps=12, dt=0.05):
        out = None
        for _ in range(steps):
            self.now += dt
            event = make_event(hand_mode=True)
            event.value["left"] = hand_flat(2.0) if left is None else left
            event.value["right"] = right
            self.emit(source, event)
            out = source.read()
        return out

    def test_open_half_closed_and_fist_map_to_0_half_and_1(self):
        source = self.source(hand_mode=True)
        self.assertAlmostEqual(self.settle(source, hand_flat(2.0)).grip, 0.0, delta=0.01)
        half = self.settle(source, hand_flat(1.475))  # halfway between 1.8 (open) and 1.15 (closed)
        self.assertAlmostEqual(half.grip, 0.5, delta=0.02)
        self.assertIn("closed", half.detail)
        self.assertAlmostEqual(self.settle(source, hand_flat(1.0)).grip, 1.0, delta=0.01)
        self.assertAlmostEqual(self.settle(source, hand_flat(1.3)).grip, 0.77, delta=0.02)
        self.assertAlmostEqual(self.settle(source, hand_flat(2.0)).grip, 0.0, delta=0.01)

    def test_gripper_follows_smoothly_and_ignores_tracking_noise(self):
        source = self.source(hand_mode=True)
        rng = np.random.default_rng(0)
        self.settle(source, hand_flat(1.475))
        values = []
        for _ in range(60):  # +-0.03 ratio noise around half closed
            values.append(self.settle(source, hand_flat(1.475 + rng.uniform(-0.03, 0.03)), steps=1, dt=0.016).grip)
        self.assertLess(max(values) - min(values), 0.08)  # the raw noise is +-0.046 in closure
        step = self.settle(source, hand_flat(1.0), steps=1, dt=0.016).grip  # a sudden fist
        self.assertLess(step, 0.65)  # does not jump straight to closed

    def test_lost_tracking_keeps_the_last_value(self):
        source = self.source(hand_mode=True)
        self.settle(source, hand_flat(1.0))
        out = self.settle(source, [0.0] * 400, steps=5)  # right hand lost: events are rejected
        self.assertAlmostEqual(out.grip, 1.0, delta=0.01)
        out = self.settle(source, hand_flat(None), steps=5)  # pose valid but finger joints not tracked
        self.assertIn("finger joints unusable", out.detail)  # falls back to the headset's own gestures
        self.assertIn("knuckles only", out.detail)

    def test_headset_gestures_are_the_fallback_when_finger_joints_are_unusable(self):
        source = self.source(hand_mode=True)
        no_joints = hand_flat(None)  # valid wrist pose, finger joints at the origin
        for state, expected in (
            ({"pinchValue": 0.0, "pinch": False}, 0.0),  # nothing recognised: stays open
            ({"pinchValue": 0.012, "pinch": True}, 1.0),  # the headset sees a pinch: closed
            ({"pinchValue": 0.05, "pinch": False}, 0.5),  # a loose pinch: half closed
            ({"pinchValue": 0.0, "pinch": False, "squeezeValue": 0.7}, 0.7),  # the headset sees a fist
        ):
            event = make_event(hand_mode=True)
            event.value["left"], event.value["right"] = hand_flat(2.0), no_joints
            event.value["rightState"] = state
            for _ in range(12):
                self.now += 0.05
                self.emit(source, event)
                out = source.read()
            self.assertAlmostEqual(out.grip, expected, delta=0.03, msg=str(state))

    def test_right_fist_is_the_gripper_not_the_clutch_and_left_fist_is_still_the_clutch(self):
        source = self.source(hand_mode=True)
        out = self.settle(source, hand_flat(1.0), left=hand_flat(2.0))
        self.assertEqual(out.clutch, 0.0)
        self.assertAlmostEqual(out.grip, 1.0, delta=0.01)
        out = self.settle(source, hand_flat(2.0), left=hand_flat(1.0))
        self.assertEqual(out.clutch, 1.0)
        self.assertAlmostEqual(out.grip, 0.0, delta=0.01)

    def test_left_pinch_no_longer_changes_the_gripper(self):
        source = self.source(hand_mode=True)
        pinch = hand_flat(finger_ratios=[1.6, 2.0, 2.0, 2.0], thumb_gap=0.012)
        out = self.settle(source, hand_flat(2.0), left=pinch)
        self.assertAlmostEqual(out.grip, 0.0, delta=0.01)

    def test_right_hand_pinch_closes_the_gripper(self):
        source = self.source(hand_mode=True)
        pinch = hand_flat(finger_ratios=[1.6, 2.0, 2.0, 2.0], thumb_gap=0.012)  # other fingers straight
        out = self.settle(source, pinch)
        self.assertAlmostEqual(out.grip, 1.0, delta=0.01)
        self.assertIn("pinch 12 mm", out.detail)
        sloppy = hand_flat(finger_ratios=[1.5, 1.6, 1.7, 1.7], thumb_gap=0.02)  # relaxed, loosely curled fingers
        self.assertAlmostEqual(self.settle(source, sloppy).grip, 1.0, delta=0.01)
        self.assertAlmostEqual(self.settle(source, hand_flat(2.0)).grip, 0.0, delta=0.01)  # released

    def test_loose_pinch_closes_partly_and_relaxed_hand_does_not_close(self):
        source = self.source(hand_mode=True)
        half = hand_flat(finger_ratios=[2.0, 2.0, 2.0, 2.0], thumb_gap=0.05)  # midway between 20 and 80 mm
        self.assertAlmostEqual(self.settle(source, half).grip, 0.5, delta=0.03)
        relaxed = hand_flat(finger_ratios=[1.95, 1.9, 1.9, 1.9], thumb_gap=0.085)  # an ordinary open hand
        self.assertAlmostEqual(self.settle(source, relaxed).grip, 0.0, delta=0.01)

    def test_thresholds_are_configurable(self):
        source = self.source(hand_mode=True, gripper_fist_open_ratio=1.6, gripper_fist_closed_ratio=1.2)
        self.assertAlmostEqual(self.settle(source, hand_flat(1.4)).grip, 0.5, delta=0.02)


if __name__ == "__main__":
    unittest.main()
