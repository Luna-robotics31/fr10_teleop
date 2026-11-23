"""The command may not run away from an arm that cannot keep up; the gripper may not delay the arm."""

import math
import threading
import time
import unittest

import numpy as np

from fr10.config import Config
from fr10.kinematics import IKSolver, RobotModel
from fr10.robots.base import RobotError
from fr10.robots.fairino import FairinoRobot
from fr10.robots.mock import MockRobot
from fr10.teleop import Mode, Teleoperator
from fr10.trajectory import JointSetpointFilter

from tests.test_fairino_gripper import FakeRPC, make_robot
from tests.test_teleop import FakeHeadset


class LeadLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg0 = Config().validate()
        cls.model = RobotModel(cls.cfg0.robot)
        cls.ik = IKSolver(cls.model, cls.cfg0.ik)

    def robot(self, lag_s, cfg=None):
        cfg = cfg or Config().validate()
        robot = MockRobot(cfg, self.model.lower, self.model.upper, lag_s=lag_s)
        robot.connect()
        self.addCleanup(robot.close)
        return robot

    def test_filter_stop_forgets_velocity(self):
        flt = JointSetpointFilter([2.0] * 2, [8.0] * 2, [-3.0] * 2, [3.0] * 2)
        flt.reset(np.zeros(2))
        for _ in range(50):
            flt.step([2.0, 2.0], 0.008)
        self.assertGreater(float(np.max(flt.velocity)), 0.5)
        flt.stop()
        np.testing.assert_array_equal(flt.velocity, np.zeros(2))

    def test_command_waits_for_an_arm_that_does_not_follow(self):
        robot = self.robot(lag_s=1e9)  # the arm never moves
        robot.start_streaming()
        target = robot.get_state().q + math.radians(60.0)
        robot.set_joint_target(target)
        time.sleep(1.0)
        lead = float(np.max(np.abs(robot.commanded_position() - robot.get_state().q)))
        self.assertLessEqual(lead, math.radians(self.cfg0.control.max_command_lead_deg) + math.radians(1.0))
        self.assertGreater(robot.lag_hold_time, 0.3)

    def test_slow_arm_is_followed_without_running_ahead_and_still_arrives(self):
        robot = self.robot(lag_s=0.25)  # a sluggish robot: 250 ms behind its command
        robot.start_streaming()
        start = robot.get_state().q.copy()
        robot.set_joint_target(start + math.radians(40.0))
        worst = 0.0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            worst = max(worst, float(np.max(np.abs(robot.commanded_position() - robot.get_state().q))))
            if np.max(np.abs(robot.get_state().q - (start + math.radians(40.0)))) < math.radians(0.5):
                break
            time.sleep(0.01)
        self.assertLessEqual(worst, math.radians(self.cfg0.control.max_command_lead_deg) + math.radians(1.0))
        np.testing.assert_allclose(robot.get_state().q, start + math.radians(40.0), atol=math.radians(0.6))
        self.assertEqual(robot.lag_hold_time, 0.0)

    def test_normal_arm_is_never_held_back(self):
        robot = self.robot(lag_s=0.03)
        robot.start_streaming()
        robot.set_joint_target(robot.get_state().q + math.radians(30.0))
        for _ in range(100):
            self.assertLess(robot.lag_hold_time, 0.05)
            time.sleep(0.02)

    def test_stuck_arm_is_a_fault_after_the_stall_timeout(self):
        cfg = Config()
        cfg.control.stall_timeout_s = 0.6
        cfg.validate()
        robot = self.robot(lag_s=0.03, cfg=cfg)
        headset = FakeHeadset()
        teleop = Teleoperator(cfg, robot, self.model, self.ik, headset)
        teleop.post("start")
        period = 1.0 / cfg.control.rate_hz
        for _ in range(6):
            teleop.step()
            time.sleep(period)
        self.assertIs(teleop.mode, Mode.READY)
        robot._lag = 1e9  # the arm stops following (e.g. protective stop)
        headset.state.clutch = 0.0
        teleop.step()
        headset.state.clutch = 1.0
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and teleop.mode is not Mode.FAULT:
            headset.move(dx=0.003)
            teleop.step()
            time.sleep(period)
        self.assertIs(teleop.mode, Mode.FAULT)
        self.assertRegex(teleop.fault_reason, "not followed|away from its command")

    def test_a_sluggish_arm_does_not_make_the_motion_stop_and_go(self):
        """Regression: a hard 6 degree limit held the command back again and again on an arm that lags 0.4 s.

        Measured with the whole pipeline (scripted operator, IK, filter): the old limit gave an acceleration
        RMS of about 880 deg/s^2 with 14 % of the ticks standing still; the soft limit gives about 290 and 0 %.
        """
        from fr10.xr_input import ScriptedInput

        class Recording(MockRobot):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sent = []

            def _send_setpoint(self, q, gripper):
                if q is not None:
                    self.sent.append((time.perf_counter(), np.array(q)))
                super()._send_setpoint(q, gripper)

        cfg = Config().validate()
        robot = Recording(cfg, self.model.lower, self.model.upper, lag_s=0.4)  # a lazy servo response
        robot.connect()
        self.addCleanup(robot.close)
        teleop = Teleoperator(
            cfg, robot, self.model, self.ik, ScriptedInput(period=6.0, start_delay=0.5, radius=0.12)
        )
        teleop.post("start")
        end = time.monotonic() + 7.5
        while time.monotonic() < end:
            teleop.step()
            time.sleep(1.0 / cfg.control.rate_hz)
        robot.close()
        times = np.array([t for t, _ in robot.sent])
        q = np.array([v for _, v in robot.sent])
        keep = times > times[0] + 1.5  # after the arm has started following
        times, q = times[keep], q[keep]
        dt = np.diff(times)
        velocity = np.diff(q, axis=0) / dt[:, None]
        acceleration = np.diff(velocity, axis=0) / dt[1:, None]
        still = float(np.mean(np.max(np.abs(velocity), axis=1) < np.deg2rad(0.5)))
        self.assertLess(np.degrees(np.sqrt(np.mean(acceleration ** 2))), 450.0)
        self.assertLess(100.0 * still, 1.0)
        soft, held = robot.lead_stats()
        self.assertLess(held, 1.0)  # never actually held back in normal motion

    def test_soft_limit_slows_gradually_and_a_blocked_arm_still_waits(self):
        robot = self.robot(lag_s=1e9)  # the arm does not move at all
        robot.start_streaming()
        robot.set_joint_target(robot.get_state().q + math.radians(70.0))
        speeds = []
        previous = robot.commanded_position()
        for _ in range(150):
            time.sleep(0.01)
            current = robot.commanded_position()
            speeds.append(float(np.max(np.abs(current - previous))) / 0.01)
            previous = current
        lead = float(np.max(np.abs(robot.commanded_position() - robot.get_state().q)))
        self.assertLessEqual(lead, math.radians(self.cfg0.control.max_command_lead_deg) + math.radians(0.5))
        self.assertGreater(lead, math.radians(self.cfg0.control.soft_command_lead_deg))
        self.assertGreater(robot.lag_hold_time, 0.2)  # blocked: it ends up waiting (the stall fault relies on this)
        self.assertLess(speeds[-1], 0.05)
        soft, held = robot.lead_stats()
        self.assertGreater(soft, 1.0)  # it was slowed on the way, not stopped dead

    def test_send_timing_is_measured(self):
        robot = self.robot(lag_s=0.03)
        robot.start_streaming()
        time.sleep(0.3)
        self.assertGreater(robot.send_max, 0.0)
        self.assertLess(robot.send_avg, 0.01)


class StrictControllerServer:
    """A local XML-RPC server that is as strict about argument types as the FAIRINO controller."""

    MOVE_TYPES = (int, int, int, int, int, int, int, float, int, int)

    def __init__(self):
        import xmlrpc.client
        from xmlrpc.server import SimpleXMLRPCServer

        self.calls = []  # ("ActGripper", args) / ("MoveGripper", args) in arrival order
        self.faults = []
        outer = self

        def check(name, given, wanted):
            for value, kind in zip(given, wanted):
                if type(value) is not kind:
                    outer.faults.append((name, given))
                    raise xmlrpc.client.Fault(
                        -501, f"Value of type {type(value).__name__.upper()} supplied where type {kind.__name__.upper()} was expected."
                    )
            outer.calls.append((name, tuple(given)))
            return 0

        self.server = SimpleXMLRPCServer(("127.0.0.1", 0), logRequests=False, allow_none=True)
        self.server.register_function(lambda *a: check("ActGripper", a, (int, int)), "ActGripper")
        self.server.register_function(lambda *a: check("MoveGripper", a, self.MOVE_TYPES), "MoveGripper")
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def gripper_robot(start_closing=0.0, **gripper_options):
    """A FairinoRobot whose gripper thread runs fast, with its controller calls recorded."""
    cfg = Config()
    cfg.fairino.gripper.min_interval_s = 0.03
    gripper_options.setdefault("mode", "open_close")  # these tests are about the open/close logic
    gripper_options.setdefault("activate_before_move", True)  # and about the plain XML-RPC path
    gripper_options.setdefault("wait_for_move", False)  # (block = 1)
    for key, value in gripper_options.items():
        setattr(cfg.fairino.gripper, key, value)
    cfg.validate()
    robot = make_robot(FakeRPC(), cfg)
    robot.sent = []
    robot.answers = {}  # method -> list of return codes to give (then 0)

    def record(method, *args, **kwargs):
        robot.sent.append((method, args))
        queue = robot.answers.get(method, [])
        return queue.pop(0) if queue else 0

    robot._rpc_call = record
    robot.set_gripper(start_closing)
    robot._streaming.set()
    robot._gripper_stop.clear()
    return robot


class GripperThread:
    """Runs a robot's gripper loop in the background for a test."""

    def __init__(self, robot, settle=0.15):
        self.robot = robot
        self.settle = settle
        self.thread = threading.Thread(target=robot._gripper_loop, daemon=True)
        robot._gripper_thread = self.thread  # what gripper_status() inspects
        robot._gripper_thread_alive_probe = self.thread.is_alive

    def __enter__(self):
        self.thread.start()
        time.sleep(self.settle)  # the start-up sync has gone out by now
        return self

    def __exit__(self, *exc):
        self.robot._gripper_stop.set()
        self.robot._streaming.clear()
        self.thread.join(timeout=2.0)


def wait_for(condition, seconds=2.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    return condition()


class GripperThreadTests(unittest.TestCase):
    def test_close_and_open_use_the_calls_of_gripper_py(self):
        robot = gripper_robot(start_closing=0.0)
        with GripperThread(robot):
            # once at the start the gripper is driven to the state it is believed to be in (open here)
            self.assertEqual(robot.sent, [("ActGripper", (2, 1)), ("MoveGripper", (2, 0, 100, 67, 2000, 1, 0, 0.0, 0, 0))])
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: len(robot.sent) >= 4))
            robot.set_gripper(0.0)
            self.assertTrue(wait_for(lambda: len(robot.sent) >= 6))
        self.assertEqual(
            robot.sent[2:6],
            [
                ("ActGripper", (2, 1)),
                ("MoveGripper", (2, 100, 100, 67, 2000, 1, 0, 0.0, 0, 0)),  # close
                ("ActGripper", (2, 1)),
                ("MoveGripper", (2, 0, 100, 67, 2000, 1, 0, 0.0, 0, 0)),  # open
            ],
        )

    def test_hysteresis_and_tracking_noise_never_send_anything(self):
        robot = gripper_robot(start_closing=0.5)
        rng = np.random.default_rng(0)
        with GripperThread(robot):
            end = time.monotonic() + 1.0
            while time.monotonic() < end:  # +-8 % noise around half closed: between 0.4 and 0.6
                robot.set_gripper(0.5 + rng.uniform(-0.08, 0.08))
                time.sleep(0.005)
        self.assertEqual(len(robot.sent), 2)  # only the start-up sync: ActGripper + MoveGripper(closed)

    def test_a_command_is_only_sent_when_the_state_changes(self):
        robot = gripper_robot(start_closing=0.0)
        with GripperThread(robot):
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: len(robot.sent) >= 4))
            time.sleep(0.4)  # the hand stays closed: no repeated commands
            robot.set_gripper(0.55)  # still above the "open" threshold: stays closed
            time.sleep(0.3)
        self.assertEqual([m for m, _ in robot.sent], ["ActGripper", "MoveGripper"] * 2)  # start-up sync + close

    def test_a_refused_command_is_retried_until_it_is_accepted(self):
        robot = gripper_robot(start_closing=0.0)
        with GripperThread(robot):
            robot.answers["MoveGripper"] = [7, 7]
            before = [m for m, _ in robot.sent].count("MoveGripper")
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: [m for m, _ in robot.sent].count("MoveGripper") >= before + 3))
            time.sleep(0.3)
        self.assertEqual([m for m, _ in robot.sent].count("MoveGripper"), before + 3)  # 2 refusals, 1 success, quiet
        self.assertEqual(robot._gripper_failures, 0)

    def test_a_refused_activation_does_not_move_the_gripper(self):
        robot = gripper_robot(start_closing=0.0)
        with GripperThread(robot):
            robot.answers["ActGripper"] = [3, 3]
            moves_before = [a for m, a in robot.sent if m == "MoveGripper"]
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: [m for m, _ in robot.sent].count("ActGripper") >= 4))  # sync + 3 tries
        moves = [a for m, a in robot.sent if m == "MoveGripper"]
        self.assertEqual(len(moves) - len(moves_before), 1)  # only after the third activation was accepted

    def test_options_wait_for_move_and_skip_activation(self):
        robot = gripper_robot(start_closing=0.0, wait_for_move=True, activate_before_move=False)
        with GripperThread(robot):
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: len(robot.sent) >= 2))
        self.assertEqual(
            robot.sent[:2],  # start-up sync (open), then close: no ActGripper, block = 0
            [("MoveGripper", (2, 0, 100, 67, 2000, 0, 0, 0.0, 0, 0)),
             ("MoveGripper", (2, 100, 100, 67, 2000, 0, 0, 0.0, 0, 0))],
        )

    def test_proportional_mode_goes_to_the_matching_position(self):
        robot = gripper_robot(start_closing=0.0, mode="proportional")
        with GripperThread(robot):
            robot.set_gripper(0.5)
            self.assertTrue(wait_for(lambda: any(m == "MoveGripper" and a[1] == 50 for m, a in robot.sent)))
        positions = [a[1] for m, a in robot.sent if m == "MoveGripper"]
        self.assertIn(50, positions)  # half closed hand: half closed gripper

    def test_gripper_commands_do_not_wait_for_the_arm_connection_lock(self):
        robot = gripper_robot(start_closing=0.0)
        with robot._rpc_lock:  # the arm's servo thread is busy in a slow ServoJ call
            with GripperThread(robot):
                robot.set_gripper(1.0)
                self.assertTrue(
                    wait_for(lambda: ("MoveGripper", (2, 100, 100, 67, 2000, 1, 0, 0.0, 0, 0)) in robot.sent),
                    "blocked by the arm's lock",
                )

    def test_the_controller_accepts_the_arguments_the_thread_sends(self):
        """Regression: an INT where the controller wants a DOUBLE made every MoveGripper fail (Fault -501)."""
        controller = StrictControllerServer()
        self.addCleanup(controller.close)
        cfg = Config()
        cfg.fairino.gripper.min_interval_s = 0.03
        cfg.fairino.gripper.mode = "open_close"
        cfg.fairino.gripper.activate_before_move = True
        cfg.fairino.gripper.wait_for_move = False
        cfg.fairino.ip = "127.0.0.1"
        cfg.fairino.rpc_port = controller.port
        robot = make_robot(FakeRPC(), cfg.validate())
        robot.set_gripper(0.0)
        robot._streaming.set()
        robot._gripper_stop.clear()
        with GripperThread(robot):
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: len(controller.calls) >= 4, 3.0))
        self.assertEqual(controller.faults, [], "the controller rejected the argument types")
        self.assertIn(("ActGripper", (2, 1)), controller.calls)
        self.assertIn(("MoveGripper", (2, 100, 100, 67, 2000, 1, 0, 0.0, 0, 0)), controller.calls)


class FakeDriver:
    """Stands in for ScriptGripper (the gripper.py calls on their own SDK connection)."""

    def __init__(self):
        self.moves, self.fail = [], 0

    def move_percent(self, percent, activate=True, block=0, log=True):
        if self.fail:
            self.fail -= 1
            raise RobotError("refused")
        self.moves.append((int(round(percent)), activate, block))

    def close_connection(self):
        self.closed = True


def driver_robot(start=0.0, **options):
    options.setdefault("mode", "proportional")
    options.setdefault("activate_before_move", False)
    robot = gripper_robot(start_closing=start, **options)
    robot._gripper_driver = FakeDriver()
    return robot


class ProportionalPinchTests(unittest.TestCase):
    """The gripper follows the pinch: half a pinch stops it halfway, a full pinch closes it."""

    def test_a_half_pinch_stops_halfway_and_a_full_pinch_closes_fully(self):
        robot = driver_robot()
        with GripperThread(robot):
            for value in (0.15, 0.3, 0.5):  # the fingers close slowly and stop halfway
                robot.set_gripper(value)
                time.sleep(0.12)
            time.sleep(0.4)  # held: no further commands
            halfway = list(robot._gripper_driver.moves)
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: robot._gripper_driver.moves[-1][0] == 100))
            robot.set_gripper(0.0)
            self.assertTrue(wait_for(lambda: robot._gripper_driver.moves[-1][0] == 0))
        self.assertEqual([m[0] for m in halfway], [0, 15, 30, 50])  # start-up sync, then it stops at 50 %
        self.assertTrue(all(p <= 50 for p, _, _ in halfway))

    def test_it_uses_the_gripper_py_connection_not_the_plain_calls(self):
        robot = driver_robot()
        with GripperThread(robot):
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 2))
        self.assertEqual(robot.sent, [], "the plain XML-RPC path must not be used")
        self.assertEqual(robot._gripper_driver.moves[0], (0, True, 1))  # start-up sync: activated, not waiting
        self.assertEqual(robot._gripper_driver.moves[1], (100, False, 1))

    def test_activation_only_before_the_first_move_never_after_a_failure(self):
        robot = driver_robot()
        with GripperThread(robot):
            robot.set_gripper(0.5)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 2))
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 3))
            robot._gripper_driver.fail = 1
            robot.set_gripper(0.0)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 4))
        activations = [activate for _, activate, _ in robot._gripper_driver.moves]
        # sync, two normal moves, and the retry after a failure: activating again takes seconds and can time out
        self.assertEqual(activations, [True, False, False, False])

    def test_noise_and_small_changes_send_nothing(self):
        robot = driver_robot(start=0.5)
        rng = np.random.default_rng(0)
        with GripperThread(robot):
            robot.set_gripper(0.5)
            time.sleep(0.2)
            sent_at_start = len(robot._gripper_driver.moves)
            end = time.monotonic() + 1.0
            while time.monotonic() < end:  # +-2 % around the held position: below the 5 % resolution
                robot.set_gripper(0.5 + rng.uniform(-0.02, 0.02))
                time.sleep(0.005)
        self.assertLessEqual(len(robot._gripper_driver.moves) - sent_at_start, 0)

    def test_wait_for_move_and_activation_options(self):
        robot = driver_robot(wait_for_move=True, activate_before_move=True)
        with GripperThread(robot):
            robot.set_gripper(0.4)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 2))
            robot.set_gripper(0.9)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 3))
        self.assertEqual(robot._gripper_driver.moves, [(0, True, 0), (40, True, 0), (90, True, 0)])  # block = 0, always activated

    def test_a_gripper_that_is_really_closed_but_reports_open_is_opened_at_start(self):
        """The reported opening was 'open' (0), the hand is open: the start-up sync must still send 'open'."""
        robot = driver_robot(start=0.0)
        with GripperThread(robot):
            pass
        self.assertEqual(robot._gripper_driver.moves[0], (0, True, 1))

    def test_resend_sends_the_present_command_again(self):
        robot = driver_robot(start=0.0)
        with GripperThread(robot):
            robot.set_gripper(0.5)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 2))
            robot.set_gripper(0.0)
            robot.resend_gripper()
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) >= 3))
        self.assertEqual([m[0] for m in robot._gripper_driver.moves], [0, 50, 0])
        self.assertTrue(robot._gripper_driver.moves[2][1])  # activated again

    def test_the_thread_reacts_to_a_new_command_at_once(self):
        """A key press or pinch must not wait for the next poll: the command goes out within a few ms."""
        robot = driver_robot(start=0.0, min_interval_s=0.5)  # a slow poll interval
        with GripperThread(robot, settle=0.7):
            before = len(robot._gripper_driver.moves)
            started = time.monotonic()
            robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: len(robot._gripper_driver.moves) > before, 2.0))
            delay = time.monotonic() - started
        self.assertLess(delay, 0.3)  # far below the 0.5 s poll interval

    def test_status_reports_thread_state_and_last_command(self):
        robot = driver_robot(start=0.0)
        robot._gripper_thread = None
        self.assertEqual(robot.gripper_status(), "GRIPPER THREAD NOT RUNNING")
        with GripperThread(robot) as running:
            running.robot.set_gripper(1.0)
            self.assertTrue(wait_for(lambda: robot._gripper_driver.moves and robot._gripper_driver.moves[-1][0] == 100))
            text = robot.gripper_status()
            self.assertIn("gripper thread running", text)
            self.assertIn("MOVE 100 %", text)
            self.assertIn("accepted", text)
        robot._gripper_driver.fail = 1000
        robot._streaming.set()  # the first run above stopped the thread: arm it again
        robot._gripper_stop.clear()
        with GripperThread(robot):
            robot.set_gripper(0.0)
            self.assertTrue(wait_for(lambda: robot._gripper_failures >= 1))
            self.assertIn("FAILED", robot.gripper_status())

    def test_a_crash_in_the_thread_is_reported_not_silent(self):
        robot = driver_robot(start=0.0)

        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        robot._send_gripper = explode
        with self.assertLogs("fr10.robots.fairino", level="ERROR") as logs:
            with GripperThread(robot):
                robot.set_gripper(1.0)
                self.assertTrue(wait_for(lambda: not robot._gripper_thread_alive_probe()))
        self.assertIn("gripper thread crashed", "\n".join(logs.output))

    def test_a_gripper_call_that_never_returns_is_reported_as_blocked(self):
        """The status must say so when the controller does not answer, not just 'no command sent yet'."""

        class BlockingDriver(FakeDriver):
            def __init__(self):
                super().__init__()
                self.release = threading.Event()

            def move_percent(self, percent, activate=True, block=0, log=True):
                self.release.wait(5.0)
                super().move_percent(percent, activate, block, log)

        robot = driver_robot(start=0.0)
        robot._gripper_driver = BlockingDriver()
        robot.GRIPPER_BUSY_REPORT_S = 0.2
        with GripperThread(robot, settle=0.6):
            self.assertIn("BLOCKED in a controller call", robot.gripper_status())
            robot._gripper_driver.release.set()
            self.assertTrue(wait_for(lambda: robot._gripper_driver.moves))
            self.assertTrue(wait_for(lambda: "BLOCKED" not in robot.gripper_status()))

    def test_no_second_sdk_connection_by_default(self):
        from types import SimpleNamespace

        def build(separate):
            created = []

            class FakeSDK:
                class RPC:
                    is_connect = True

                    def __init__(self, ip):
                        created.append(ip)
                        self.calls = []

                    def ActGripper(self, *a):
                        return 0

                    def MoveGripper(self, *a):
                        return 0

                    def CloseRPC(self):
                        pass

            cfg = Config()
            cfg.fairino.gripper.use_process = False  # these tests are about the thread path
            cfg.fairino.gripper.separate_sdk_connection = separate
            robot = make_robot(FakeRPC(), cfg.validate())
            robot._rpc.robot_state_pkg = SimpleNamespace(gripper_active=1 << 2, gripper_position=0)
            robot._sdk = FakeSDK
            robot._prepare_gripper()
            robot._gripper_stop.set()
            robot._gripper_wake.set()
            robot._gripper_thread.join(timeout=2.0)
            return created, robot

        created, robot = build(False)
        self.assertEqual(created, [], "the gripper must not open a second vendor-SDK connection by default")
        self.assertIsNone(robot._gripper_driver)
        created, robot = build(True)
        self.assertEqual(len(created), 1)
        self.assertIsNotNone(robot._gripper_driver)

    def test_the_connection_is_closed_on_disconnect(self):
        robot = driver_robot()
        driver = robot._gripper_driver
        robot._rpc = None
        robot._disconnect()
        self.assertTrue(driver.closed)


if __name__ == "__main__":
    unittest.main()
