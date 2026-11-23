"""Integration tests of the teleoperation state machine with the in-process mock robot.

These run the real control code in real time (servo thread at 125 Hz, IK loop at 60 Hz);
only the robot and the headset are stand-ins.
"""

import json
import math
import tempfile
import time
import unittest

import numpy as np

from fr10.config import Config
from fr10.kinematics import IKSolver, RobotModel
from fr10.mapping import PALM_DOWN
from fr10.math_utils import make_pose, pose_error, rot_z
from fr10.recorder import EpisodeRecorder
from fr10.robots.base import RobotError
from fr10.robots.mock import MockRobot
from fr10.teleop import Mode, Teleoperator
from fr10.xr_input import InputSource, OperatorInput, ScriptedInput

HAND = make_pose(PALM_DOWN["right"], [0.3, -0.2, 1.1])  # palm down, fingers forward
HEAD = make_pose(np.eye(3), [0.0, 0.0, 1.6])


class FakeHeadset(InputSource):
    """Operator input that the test sets directly."""

    def __init__(self):
        self.state = OperatorInput(connected=True, valid=True, hand_pose=HAND.copy(), head_pose=HEAD.copy())
        self.shown = 0

    def read(self) -> OperatorInput:
        s = self.state
        return OperatorInput(s.connected, s.valid, None if s.hand_pose is None else s.hand_pose.copy(),
                             s.head_pose, s.clutch, s.grip, s.button_a, s.button_b)

    def move(self, dx=0.0, dy=0.0, dz=0.0, yaw=0.0):
        pose = self.state.hand_pose.copy()
        pose[:3, 3] += [dx, dy, dz]
        pose[:3, :3] = rot_z(yaw) @ pose[:3, :3]
        self.state.hand_pose = pose


class TeleopTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_cfg = Config().validate()
        cls.model = RobotModel(cls.base_cfg.robot)
        cls.ik = IKSolver(cls.model, cls.base_cfg.ik)

    def make(self, cfg=None, recorder=None):
        cfg = cfg or Config().validate()
        self.cfg = cfg
        self.robot = MockRobot(cfg, self.model.lower, self.model.upper)
        self.robot.connect()
        self.addCleanup(self.robot.close)
        self.headset = FakeHeadset()
        self.teleop = Teleoperator(cfg, self.robot, self.model, self.ik, self.headset, recorder=recorder)
        return self.teleop

    def spin(self, seconds, until=None):
        """Run control cycles at the configured rate for ``seconds`` (or until ``until()`` is true)."""
        period = 1.0 / self.cfg.control.rate_hz
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.teleop.step()
            if until is not None and until():
                return True
            time.sleep(period)
        return until is None

    def start(self):
        self.teleop.post("start")
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.READY)

    def engage(self):
        self.headset.state.clutch = 0.0
        self.spin(0.05)
        self.headset.state.clutch = 1.0
        self.spin(0.05)
        self.assertIs(self.teleop.mode, Mode.TRACKING)

    def tcp(self):
        return self.model.fk(self.robot.get_state().q)


class StateMachineTests(TeleopTestCase):
    def test_idle_sends_nothing(self):
        self.make()
        self.headset.state.clutch = 1.0
        self.spin(0.3)
        self.assertIs(self.teleop.mode, Mode.IDLE)
        self.assertFalse(self.robot.streaming)
        self.assertEqual(self.robot.setpoints_received, 0)

    def test_start_holds_position_and_needs_clutch_release_first(self):
        self.make()
        q0 = self.robot.get_state().q
        self.headset.state.clutch = 1.0  # operator already squeezing when 'r' is pressed
        self.start()
        self.assertTrue(self.robot.streaming)
        self.headset.move(dx=0.1)
        self.spin(0.4)
        self.assertIs(self.teleop.mode, Mode.READY)  # not engaged: clutch was never open
        np.testing.assert_allclose(self.robot.get_state().q, q0, atol=1e-6)
        self.engage()  # release + squeeze engages

    def test_tool_follows_hand_translation_and_rotation(self):
        self.make()
        self.start()
        start_tcp = self.tcp()
        self.engage()
        for _ in range(30):  # 10 cm forward, 6 cm left, 5 cm up and 0.3 rad yaw over 0.5 s
            self.headset.move(dx=0.10 / 30, dy=0.06 / 30, dz=0.05 / 30, yaw=0.3 / 30)
            self.spin(1 / 60)
        self.spin(1.5)
        expected = start_tcp.copy()
        expected[:3, 3] += [0.10, 0.06, 0.05]
        expected[:3, :3] = rot_z(0.3) @ start_tcp[:3, :3]
        position_error, rotation_error = pose_error(self.tcp(), expected)
        self.assertLess(position_error, 2e-3)
        self.assertLess(rotation_error, math.radians(0.5))

    def test_engaging_needs_the_hand_to_match_the_gripper(self):
        """Palm sideways: the clutch does not engage (no swing); palm down: it does and the gripper stays down."""
        self.make()
        self.start()
        q0 = self.robot.get_state().q.copy()
        sideways = HAND.copy()
        sideways[:3, :3] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])  # palm facing right, not down
        self.headset.state.hand_pose = sideways
        self.headset.state.clutch = 0.0
        self.spin(0.1)
        self.headset.state.clutch = 1.0
        self.spin(0.6)
        self.assertIs(self.teleop.mode, Mode.READY)
        np.testing.assert_allclose(self.robot.get_state().q, q0, atol=1e-6)
        self.headset.state.hand_pose = HAND.copy()  # turn the palm down while still squeezing
        self.spin(0.3)
        self.assertIs(self.teleop.mode, Mode.TRACKING)
        np.testing.assert_allclose(self.tcp()[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)
        for _ in range(30):  # move the hand around: with the palm kept down the gripper keeps pointing down
            self.headset.move(dx=0.003, dy=0.002, dz=-0.001)
            self.spin(1 / 60)
        self.spin(0.5)
        np.testing.assert_allclose(self.tcp()[:3, 2], [0.0, 0.0, -1.0], atol=2e-3)

    def test_clutch_release_freezes_arm_and_reengage_does_not_jump(self):
        self.make()
        self.start()
        self.engage()
        for _ in range(20):
            self.headset.move(dx=0.004)
            self.spin(1 / 60)
        self.headset.state.clutch = 0.0
        self.spin(1.0)
        self.assertIs(self.teleop.mode, Mode.READY)
        held = self.robot.get_state().q
        self.headset.move(dx=-0.3, dy=0.4, dz=-0.2)  # hand repositions freely (palm stays down)
        self.spin(0.5)
        np.testing.assert_allclose(self.robot.get_state().q, held, atol=1e-5)
        self.headset.state.clutch = 1.0
        self.spin(0.5)
        self.assertIs(self.teleop.mode, Mode.TRACKING)
        np.testing.assert_allclose(self.robot.get_state().q, held, atol=1e-3)

    def test_joint_speed_limit_is_respected_for_violent_hand_motion(self):
        self.make()
        self.start()
        self.engage()
        limit = np.asarray(self.cfg.robot.max_velocity_rad_s)
        previous, previous_time, worst = self.robot.commanded_position(), time.monotonic(), np.zeros(6)
        for k in range(90):
            self.headset.move(dx=0.1 * math.sin(k), dy=0.1 * math.cos(1.3 * k), dz=0.05 * math.sin(0.7 * k))
            self.teleop.step()
            time.sleep(1 / 60)
            now, current = time.monotonic(), self.robot.commanded_position()
            worst = np.maximum(worst, np.abs(current - previous) / (now - previous_time))
            previous, previous_time = current, now
        self.assertTrue(np.all(worst <= limit * 1.25), f"{worst} > {limit}")  # 25 % allows for timing jitter
        self.assertTrue(self.model.within_limits(self.robot.commanded_position(), 1e-9))

    def test_hand_jump_drops_clutch(self):
        self.make()
        self.start()
        self.engage()
        held = self.robot.get_state().q
        self.headset.move(dx=0.6)  # tracking glitch
        self.spin(0.5)
        self.assertIs(self.teleop.mode, Mode.READY)
        np.testing.assert_allclose(self.robot.get_state().q, held, atol=1e-4)
        self.spin(0.3)  # still squeezing: must NOT re-engage on its own
        self.assertIs(self.teleop.mode, Mode.READY)
        self.engage()

    def test_tracking_loss_drops_clutch(self):
        self.make()
        self.start()
        self.engage()
        self.headset.state.valid = False
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.READY)
        self.headset.state.valid = True
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.READY)

    def test_workspace_floor_is_enforced(self):
        self.make()
        self.start()
        self.engage()
        for _ in range(120):
            self.headset.move(dz=-0.01)  # push 1.2 m down, through the table
            self.spin(1 / 60)
        self.spin(1.5)
        self.assertGreaterEqual(self.tcp()[2, 3], self.cfg.mapping.workspace.z[0] - 2e-3)
        self.assertTrue(self.teleop.workspace_limited)

    def test_home(self):
        cfg = Config()
        cfg.control.home_speed_deg_s = 60.0
        self.make(cfg.validate())
        self.start()
        self.engage()
        for _ in range(30):
            self.headset.move(dy=0.005, dz=0.004)
            self.spin(1 / 60)
        self.teleop.post("home")
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.HOMING)  # squeezed clutch does not cancel until released once
        self.assertTrue(self.spin(8.0, until=lambda: self.teleop.mode is Mode.READY))
        self.spin(0.8)
        np.testing.assert_allclose(self.robot.get_state().q, self.model.home, atol=2e-3)

    def test_home_on_start_moves_to_ready_pose_then_waits_for_the_clutch(self):
        cfg = Config()
        cfg.control.home_on_start = True
        cfg.control.home_speed_deg_s = 90.0
        self.make(cfg.validate())
        displaced = self.model.home.copy()
        displaced[0] += math.radians(30.0)
        displaced[2] -= math.radians(25.0)
        self.robot.set_joint_positions(displaced)
        self.headset.state.clutch = 0.0
        self.teleop.post("start")
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.HOMING)
        self.assertTrue(self.spin(8.0, until=lambda: self.teleop.mode is Mode.READY))
        self.spin(0.8)
        np.testing.assert_allclose(self.robot.get_state().q, self.model.home, atol=2e-3)
        self.spin(0.3)
        self.assertIs(self.teleop.mode, Mode.READY)  # waits: nothing moves until the clutch closes
        self.engage()

    def test_default_start_homes_first_unless_already_there(self):
        self.make()  # default configuration: control.home_on_start is on
        self.assertTrue(self.cfg.control.home_on_start)
        self.teleop.post("start")
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.READY)  # already at home: no move
        self.teleop.post("stop")
        self.spin(0.1)
        displaced = self.model.home.copy()
        displaced[0] += math.radians(40.0)
        self.robot.set_joint_positions(displaced)
        self.headset.state.clutch = 0.0
        self.teleop.post("start")
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.HOMING)
        self.assertTrue(self.spin(15.0, until=lambda: self.teleop.mode is Mode.READY))
        self.spin(1.0)
        np.testing.assert_allclose(self.robot.get_state().q, self.model.home, atol=2e-3)

    def test_no_home_on_start_holds_the_arm_where_it_is(self):
        cfg = Config()
        cfg.control.home_on_start = False
        self.make(cfg.validate())
        displaced = self.model.home.copy()
        displaced[0] += math.radians(40.0)
        self.robot.set_joint_positions(displaced)
        self.teleop.post("start")
        self.spin(0.3)
        self.assertIs(self.teleop.mode, Mode.READY)
        np.testing.assert_allclose(self.robot.get_state().q, displaced, atol=1e-4)

    def test_home_is_cancelled_by_clutch(self):
        self.make()
        self.start()
        self.engage()
        for _ in range(30):
            self.headset.move(dy=0.006)
            self.spin(1 / 60)
        self.headset.state.clutch = 0.0
        self.headset.state.button_b = True  # controller B button
        self.spin(0.1)
        self.headset.state.button_b = False
        self.assertIs(self.teleop.mode, Mode.HOMING)
        self.headset.state.clutch = 1.0
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.READY)
        self.assertGreater(np.max(np.abs(self.robot.get_state().q - self.model.home)), 1e-2)


class HandModeTeleopTests(TeleopTestCase):
    """Hand tracking: the clutch value comes from the left fist, everything else as with controllers."""

    def make_hand(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        self.make(cfg.validate())
        self.start()

    def test_fist_engages_follows_and_open_hand_holds(self):
        self.make_hand()
        start_tcp = self.tcp()
        self.headset.state.clutch = 0.0
        self.spin(0.1)
        self.headset.move(dx=0.3)  # hand far away: nothing happens without the fist
        self.spin(0.3)
        self.assertIs(self.teleop.mode, Mode.READY)
        self.headset.state.clutch = 1.0  # left fist
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.TRACKING)
        for _ in range(30):
            self.headset.move(dx=0.004, dz=0.002)
            self.spin(1 / 60)
        self.spin(1.0)
        np.testing.assert_allclose(self.tcp()[:3, 3] - start_tcp[:3, 3], [0.12, 0.0, 0.06], atol=3e-3)
        self.headset.state.clutch = 0.0  # open the left hand
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.READY)

    def test_fist_held_while_pressing_r_does_not_move_the_arm(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        self.make(cfg.validate())
        self.headset.state.clutch = 1.0
        self.teleop.post("start")
        self.spin(0.5)
        self.assertIs(self.teleop.mode, Mode.READY)  # the left hand must be opened once first
        self.headset.state.clutch = 0.0
        self.spin(0.1)
        self.headset.state.clutch = 1.0
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.TRACKING)

    def test_tracking_loss_needs_the_hand_to_be_opened_again(self):
        self.make_hand()
        self.headset.state.clutch = 0.0
        self.spin(0.1)
        self.headset.state.clutch = 1.0
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.TRACKING)
        self.headset.state.valid = False
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.READY)
        self.headset.state.valid = True
        self.spin(0.4)
        self.assertIs(self.teleop.mode, Mode.READY)  # fist still closed: does not resume by itself
        self.headset.state.clutch = 0.0
        self.spin(0.1)
        self.headset.state.clutch = 1.0
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.TRACKING)


class HandGripperFistTests(TeleopTestCase):
    """Default hand-mode gripper: the closure of the right hand sets the opening."""

    def make_hand(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        self.make(cfg.validate())

    def test_gripper_follows_the_hand_closure(self):
        self.make_hand()
        self.headset.state.grip = 0.0
        self.start()
        self.spin(0.3)
        for closure in (0.5, 1.0, 0.25, 0.0):
            self.headset.state.grip = closure
            self.spin(1.0)
            self.assertAlmostEqual(self.robot.get_state().gripper, closure, delta=0.05)

    def test_starting_never_opens_a_gripper_that_holds_a_part(self):
        self.make_hand()
        self.robot._gripper = 1.0  # closed on an object
        self.robot.set_gripper(1.0)
        self.headset.state.grip = 0.0  # but the operator's hand is open when pressing r
        self.start()
        self.spin(1.0)
        self.assertAlmostEqual(self.robot.get_state().gripper, 1.0, delta=0.02)  # still closed
        self.headset.state.grip = 0.9  # the hand closes to match: control passes to the hand
        self.spin(0.5)
        self.headset.state.grip = 0.3
        self.spin(1.2)
        self.assertAlmostEqual(self.robot.get_state().gripper, 0.3, delta=0.05)


    def test_a_gripper_whose_reported_opening_is_wrong_is_not_locked_for_ever(self):
        """The real gripper may report an opening that matches neither an open nor a closed hand."""
        self.make_hand()
        self.robot._gripper = 0.5
        self.robot.set_gripper(0.5)
        self.headset.state.grip = 0.0  # open hand: does not match the reported 50 %
        self.start()
        self.spin(0.6)
        self.assertAlmostEqual(self.robot.get_state().gripper, 0.5, delta=0.02)  # held, not driven
        self.headset.state.grip = 1.0  # a clear pinch / fist: control passes to the hand
        self.spin(1.5)
        self.assertGreater(self.robot.get_state().gripper, 0.95)
        self.headset.state.grip = 0.0
        self.spin(1.5)
        self.assertLess(self.robot.get_state().gripper, 0.05)


class HandPublisherTests(TeleopTestCase):
    def test_every_cycle_offers_the_headset_to_the_publisher(self):
        calls = []

        class Publisher:
            def update(self, source, now):
                calls.append((source, now))

        cfg = Config()
        cfg.xr.input_mode = "hand"
        self.make(cfg.validate())
        self.teleop.hand_publisher = Publisher()
        self.spin(0.2)
        self.assertGreater(len(calls), 3)
        self.assertIs(calls[0][0], self.headset)


class OpenGripperKeyTests(TeleopTestCase):
    def test_o_opens_the_gripper_and_asks_the_robot_to_resend(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        self.make(cfg.validate())
        resent = []
        self.robot.resend_gripper = lambda: resent.append(True)
        self.robot._gripper = 1.0  # closed
        self.robot.set_gripper(1.0)
        self.headset.state.grip = 0.0
        self.start()
        self.teleop.post("open_gripper")
        self.spin(1.5)
        self.assertEqual(resent, [True])
        self.assertLess(self.robot.get_state().gripper, 0.05)

    def test_g_closes_the_gripper_and_an_open_hand_does_not_undo_it(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        self.make(cfg.validate())
        resent = []
        self.robot.resend_gripper = lambda: resent.append(True)
        self.headset.state.grip = 0.0  # open hand
        self.start()
        self.teleop.post("close_gripper")
        self.spin(1.5)
        self.assertEqual(resent, [True])
        self.assertGreater(self.robot.get_state().gripper, 0.95)  # closed, although the hand is open
        self.headset.state.grip = 0.9  # the hand is clearly closed: control passes to it
        self.spin(0.5)
        self.headset.state.grip = 0.0  # and now opening the hand opens the gripper
        self.spin(1.5)
        self.assertLess(self.robot.get_state().gripper, 0.05)

    def test_o_overrides_a_closed_hand_until_the_hand_moves(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        self.make(cfg.validate())
        self.headset.state.grip = 1.0  # closed hand
        self.start()
        self.spin(0.5)
        self.teleop.post("open_gripper")
        self.spin(1.5)
        self.assertLess(self.robot.get_state().gripper, 0.05)  # open although the hand is closed
        self.headset.state.grip = 0.0
        self.spin(0.3)
        self.headset.state.grip = 1.0
        self.spin(1.5)
        self.assertGreater(self.robot.get_state().gripper, 0.95)  # the hand has taken over again

    def test_g_before_r_does_nothing(self):
        self.make()
        resent = []
        self.robot.resend_gripper = lambda: resent.append(True)
        self.teleop.post("close_gripper")
        self.spin(0.1)
        self.assertEqual(resent, [])

    def test_o_before_r_does_nothing(self):
        self.make()
        resent = []
        self.robot.resend_gripper = lambda: resent.append(True)
        self.teleop.post("open_gripper")
        self.spin(0.1)
        self.assertEqual(resent, [])


class HandGripperToggleTests(TeleopTestCase):
    def test_each_pinch_toggles_the_gripper_in_hand_mode(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        cfg.xr.gripper_source = "both_hands_pinch"
        self.make(cfg.validate())
        self.start()
        for expected in (1.0, 0.0, 1.0):
            self.headset.state.grip = 1.0  # a pinch of the left hand
            self.spin(0.1)
            self.headset.state.grip = 0.0
            self.spin(1.0)
            self.assertAlmostEqual(self.robot.get_state().gripper, expected, delta=0.05)

    def test_holding_a_pinch_does_not_keep_toggling(self):
        cfg = Config()
        cfg.xr.input_mode = "hand"
        cfg.xr.gripper_source = "both_hands_pinch"
        self.make(cfg.validate())
        self.start()
        self.headset.state.grip = 1.0
        self.spin(1.5)
        self.assertGreater(self.robot.get_state().gripper, 0.95)  # closed once and stays closed


class SafetyTests(TeleopTestCase):
    def test_robot_fault_stops_streaming_and_recovers(self):
        self.make()
        self.start()
        self.engage()
        self.robot.inject_fault("controller error 1/33")
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.FAULT)
        self.assertIn("1/33", self.teleop.fault_reason)
        self.assertFalse(self.robot.streaming)
        count = self.robot.setpoints_received
        self.headset.move(dx=0.1)
        self.spin(0.3)
        self.assertEqual(self.robot.setpoints_received, count)  # nothing is sent while faulted
        self.teleop.post("start")  # clears the fault
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.READY)
        self.assertTrue(self.robot.streaming)

    def test_stale_feedback_is_a_fault(self):
        cfg = Config()
        cfg.control.state_timeout_s = 0.2
        self.make(cfg.validate())
        self.start()
        self.robot.freeze_feedback(True)
        self.spin(0.6)
        self.assertIs(self.teleop.mode, Mode.FAULT)
        self.assertIn("feedback", self.teleop.fault_reason)
        self.teleop.post("start")  # still frozen: refuses to start
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.FAULT)
        self.robot.freeze_feedback(False)
        self.teleop.post("start")
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.READY)

    def test_arm_not_following_is_a_fault(self):
        cfg = Config()
        cfg.control.home_on_start = False  # this test is about holding the arm where it is
        self.make(cfg.validate())
        self.start()
        displaced = self.robot.get_state().q.copy()
        displaced[0] += math.radians(40.0)  # e.g. pushed away / protective stop
        self.robot.set_joint_positions(displaced)
        self.robot.inject_fault(None)
        self.robot._lag = 1e9  # the arm no longer follows set-points
        self.spin(1.0)
        self.assertIs(self.teleop.mode, Mode.FAULT)
        self.assertIn("away from its command", self.teleop.fault_reason)
        self.robot._lag = 0.03
        self.teleop.post("start")
        self.spin(0.3)
        self.assertIs(self.teleop.mode, Mode.READY)
        # recovery restarts from where the arm actually is: no jump back
        np.testing.assert_allclose(self.robot.get_state().q, displaced, atol=1e-3)

    def test_start_refused_outside_joint_range(self):
        self.make()
        bad = self.model.home.copy()
        bad[1] = self.model.upper[1] + 0.2
        self.robot.set_joint_positions(bad)
        self.teleop.post("start")
        self.spin(0.1)
        self.assertIs(self.teleop.mode, Mode.IDLE)
        self.assertFalse(self.robot.streaming)
        with self.assertRaises(RobotError):
            self.robot.start_streaming()

    def test_quit_stops_everything(self):
        self.make()
        self.start()
        self.teleop.post("quit")
        self.teleop.run()  # returns immediately and shuts down
        self.assertTrue(self.teleop.stopped)
        self.assertFalse(self.robot.streaming)
        with self.assertRaises(ValueError):
            self.teleop.post("explode")

    def test_stop_halts_arm_at_once_and_r_resumes(self):
        cfg = Config()
        cfg.control.home_on_start = False  # this test is about holding the arm where it is
        self.make(cfg.validate())
        self.start()
        self.engage()
        for _ in range(20):
            self.headset.move(dx=0.004)
            self.spin(1 / 60)
        self.teleop.post("stop")
        self.spin(0.05)
        self.assertIs(self.teleop.mode, Mode.IDLE)
        self.assertFalse(self.robot.streaming)
        count, held = self.robot.setpoints_received, self.robot.get_state().q.copy()
        self.headset.move(dx=0.1)  # still squeezing and moving: nothing may happen
        self.spin(0.4)
        self.assertEqual(self.robot.setpoints_received, count)
        np.testing.assert_allclose(self.robot.get_state().q, held, atol=1e-6)
        self.teleop.post("stop")  # harmless when already stopped
        self.teleop.post("start")
        self.spin(0.2)
        self.assertIs(self.teleop.mode, Mode.READY)  # holding the clutch does not move the arm
        np.testing.assert_allclose(self.robot.get_state().q, held, atol=1e-4)
        self.engage()

    def test_speed_scale_events(self):
        self.make()
        self.teleop.post("slower")
        self.teleop.post("slower")
        self.spin(0.05)
        self.assertAlmostEqual(self.robot.speed_scale, 0.8)
        for _ in range(12):
            self.teleop.post("slower")
        self.spin(0.05)
        self.assertAlmostEqual(self.robot.speed_scale, 0.1)


class GripperTests(TeleopTestCase):
    def test_analog_gripper_follows_trigger_and_survives_dropout(self):
        self.make()
        self.start()
        self.headset.state.grip = 1.0
        self.spin(1.2)
        self.assertGreater(self.robot.get_state().gripper, 0.95)
        self.headset.state.valid = False  # controller drops out with the part in hand
        self.headset.state.grip = 0.0
        self.spin(0.5)
        self.assertGreater(self.robot.get_state().gripper, 0.95)
        self.headset.state.valid = True
        self.spin(1.2)
        self.assertLess(self.robot.get_state().gripper, 0.05)

    def test_toggle_gripper(self):
        cfg = Config()
        cfg.gripper.mode = "toggle"
        self.make(cfg.validate())
        self.start()
        for expected in (1.0, 0.0, 1.0):
            self.headset.state.grip = 1.0
            self.spin(0.1)
            self.headset.state.grip = 0.0
            self.spin(1.0)
            self.assertAlmostEqual(self.robot.get_state().gripper, expected, delta=0.05)

    def test_gripper_disabled(self):
        cfg = Config()
        cfg.gripper.enabled = False
        self.make(cfg.validate())
        self.start()
        self.headset.state.grip = 1.0
        self.spin(0.5)
        self.assertEqual(self.robot.get_state().gripper, 0.0)


class RecordingAndScriptTests(TeleopTestCase):
    def test_recording_writes_an_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config()
            cfg.recording.directory = tmp
            cfg.recording.rate_hz = 30.0
            cfg.validate()
            recorder = EpisodeRecorder(cfg, "mock")
            self.make(cfg, recorder)
            self.start()
            self.headset.state.button_a = True  # controller A toggles recording
            self.spin(0.05)
            self.headset.state.button_a = False
            self.assertTrue(recorder.active)
            self.engage()
            for _ in range(60):
                self.headset.move(dx=0.001)
                self.spin(1 / 60)
            self.teleop.post("record")
            self.spin(0.05)
            self.assertFalse(recorder.active)
            episode = recorder._root / "episode_0001"
            steps = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
            meta = json.loads((episode / "meta.json").read_text())
            self.assertEqual(meta["joint_names"], ["j1", "j2", "j3", "j4", "j5", "j6"])
            self.assertTrue(25 <= len(steps) <= 60, len(steps))  # 1.2 - 1.6 s at 30 Hz
            self.assertEqual([s["step"] for s in steps], list(range(len(steps))))
            for key in ("q", "dq", "q_command", "q_target"):
                self.assertEqual(len(steps[-1][key]), 6)
            self.assertEqual(len(steps[-1]["tcp_pose"]), 7)
            self.assertIn(steps[-1]["mode"], ("tracking", "ready"))
            recorder.start()
            self.assertEqual(recorder.episode_dir.name, "episode_0002")
            self.teleop.shutdown()  # closes the open episode
            self.assertFalse(recorder.active)

    def test_scripted_operator_runs_the_whole_pipeline(self):
        cfg = Config().validate()
        self.cfg = cfg
        robot = MockRobot(cfg, self.model.lower, self.model.upper)
        robot.connect()
        self.addCleanup(robot.close)
        teleop = Teleoperator(cfg, robot, self.model, self.ik, ScriptedInput(period=4.0, start_delay=0.2))
        self.teleop = teleop
        teleop.post("start")
        tcp0 = self.model.fk(robot.get_state().q)
        travelled, last = 0.0, tcp0[:3, 3]
        deadline = time.monotonic() + 4.4
        while time.monotonic() < deadline:
            teleop.step()
            time.sleep(1 / 60)
            p = self.model.fk(robot.get_state().q)[:3, 3]
            travelled += float(np.linalg.norm(p - last))
            last = p
        self.assertIs(teleop.mode, Mode.TRACKING)
        self.assertGreater(travelled, 0.5)  # circle of 0.12 m radius: ~0.75 m path
        self.assertLess(float(np.linalg.norm(last - tcp0[:3, 3])), 0.05)  # back near the start after one lap
        self.assertEqual(self.ik.failures, 0)
        self.assertLess(robot.max_lateness, 0.02)
        teleop.shutdown()


if __name__ == "__main__":
    unittest.main()
