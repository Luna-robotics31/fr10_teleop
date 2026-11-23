"""The gripper in a process of its own: the loop logic, a really spawned process, and the robot backend."""

import threading
import time
import unittest
from types import SimpleNamespace

from fr10.config import Config
from fr10.gripper_process import (
    COMMANDS, FAILURES, STATE, STATE_FAILED, STATE_RUNNING, STATUS_LEN, GripperProcess, gripper_main,
)
from fr10.robots.base import RobotError
from fr10.robots.fairino import FairinoRobot
from tests.test_command_lead import StrictControllerServer, wait_for
from tests.test_fairino_gripper import FakeRPC


class FakeGripper:
    def __init__(self):
        self.activations, self.moves, self.fail_moves, self.fail_activation = 0, [], 0, False
        self.activate_each_move, self.block, self.closed = False, 1, False

    def activate(self):
        self.activations += 1
        if self.fail_activation:
            raise RobotError("activation timed out")

    def move_percent(self, percent, activate=None, block=None, log=True):
        if self.fail_moves:
            self.fail_moves -= 1
            raise RobotError("refused")
        self.moves.append((int(round(percent)), self.activate_each_move if activate is None else activate,
                           self.block if block is None else block))

    def close_connection(self):
        self.closed = True


class LoopHarness:
    """Runs gripper_main in a thread, with plain Python stand-ins for the shared objects."""

    def __init__(self, cfg=None, gripper=None, closure=0.0, min_interval=0.05):
        self.cfg = cfg or Config().validate()
        self.cfg.fairino.gripper.min_interval_s = min_interval
        self.gripper = gripper or FakeGripper()
        self.target = SimpleNamespace(value=closure)
        self.status = [0.0] * STATUS_LEN
        self.resend, self.stop, self.ready = threading.Event(), threading.Event(), threading.Event()
        self.thread = threading.Thread(
            target=gripper_main,
            args=(self.cfg, self.target, self.status, self.resend, self.stop, self.ready, lambda cfg: self.gripper),
            daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(3.0)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=3.0)


class LoopTests(unittest.TestCase):
    def test_activates_once_then_follows_the_target(self):
        with LoopHarness(closure=0.0) as h:
            self.assertTrue(wait_for(lambda: h.gripper.moves))  # the start-up sync
            h.target.value = 1.0
            self.assertTrue(wait_for(lambda: h.gripper.moves[-1][0] == 100))
            h.target.value = 0.0
            self.assertTrue(wait_for(lambda: h.gripper.moves[-1][0] == 0))
        self.assertEqual(h.gripper.activations, 1)  # activated once, never again
        self.assertTrue(all(activate is False and block == 1 for _, activate, block in h.gripper.moves))
        self.assertTrue(h.gripper.closed)  # the connection is closed when the process ends

    def test_half_closure_stops_halfway_with_few_commands(self):
        with LoopHarness(closure=0.0, min_interval=0.4) as h:  # the real default spacing
            time.sleep(0.2)
            before = len(h.gripper.moves)
            for value in (0.1, 0.2, 0.3, 0.4, 0.5):  # the fingers close and stop halfway
                h.target.value = value
                time.sleep(0.12)
            time.sleep(1.2)
            sent = h.gripper.moves[before:]
            count = len(h.gripper.moves)
            time.sleep(0.5)  # held: nothing more
            self.assertEqual(len(h.gripper.moves), count)
        self.assertLessEqual(len(sent), 4)
        self.assertAlmostEqual(sent[-1][0], 50, delta=3)

    def test_resend_sends_the_present_position_again(self):
        with LoopHarness(closure=0.5) as h:
            self.assertTrue(wait_for(lambda: len(h.gripper.moves) >= 1))
            count = len(h.gripper.moves)
            h.resend.set()
            self.assertTrue(wait_for(lambda: len(h.gripper.moves) > count))
        self.assertEqual(h.gripper.moves[-1][0], 50)

    def test_refused_moves_are_retried_and_counted(self):
        gripper = FakeGripper()
        gripper.fail_moves = 2
        with LoopHarness(gripper=gripper, closure=1.0) as h:
            self.assertTrue(wait_for(lambda: h.gripper.moves, 4.0))
            self.assertEqual(h.status[COMMANDS], 1)
            self.assertEqual(h.status[FAILURES], 0)  # reset by the success that followed the refusals
        self.assertEqual(h.gripper.moves[0][0], 100)

    def test_a_failed_activation_does_not_stop_the_process(self):
        gripper = FakeGripper()
        gripper.fail_activation = True
        with LoopHarness(gripper=gripper, closure=1.0) as h:
            self.assertTrue(wait_for(lambda: h.gripper.moves))
            self.assertEqual(h.status[STATE], STATE_RUNNING)

    def test_a_controller_that_cannot_be_reached_is_reported(self):
        def broken(cfg):
            raise RobotError("no answer")

        cfg = Config().validate()
        status, ready = [0.0] * STATUS_LEN, threading.Event()
        gripper_main(cfg, SimpleNamespace(value=0.0), status, threading.Event(), threading.Event(), ready, broken)
        self.assertTrue(ready.is_set())  # the arm must not wait for ever for a gripper that cannot start
        self.assertEqual(status[STATE], STATE_FAILED)

    def test_open_close_mode_only_goes_to_the_ends_with_a_gap(self):
        cfg = Config()
        cfg.fairino.gripper.mode = "open_close"
        with LoopHarness(cfg=cfg.validate(), closure=0.0) as h:
            self.assertTrue(wait_for(lambda: h.gripper.moves))
            for value in (0.2, 0.5, 0.55):  # between the thresholds: stays open
                h.target.value = value
                time.sleep(0.15)
            self.assertEqual({p for p, _, _ in h.gripper.moves}, {0})
            h.target.value = 0.7
            self.assertTrue(wait_for(lambda: h.gripper.moves[-1][0] == 100))
            h.target.value = 0.5  # still above the "open" threshold: stays closed
            time.sleep(0.4)
            self.assertEqual(h.gripper.moves[-1][0], 100)
            h.target.value = 0.2
            self.assertTrue(wait_for(lambda: h.gripper.moves[-1][0] == 0))


def process_config(controller_port):
    cfg = Config()
    cfg.fairino.ip = "127.0.0.1"
    cfg.fairino.rpc_port = controller_port
    cfg.fairino.rpc_timeout_s = 2.0
    cfg.fairino.gripper.min_interval_s = 0.05
    return cfg.validate()


class RealProcessTests(unittest.TestCase):
    def test_a_spawned_process_drives_a_type_strict_controller(self):
        controller = StrictControllerServer()
        self.addCleanup(controller.close)
        process = GripperProcess(process_config(controller.port), initial_closure=0.0)
        process.start()
        self.addCleanup(process.stop)
        self.assertTrue(process.wait_ready(20.0), "the process did not finish activating")
        self.assertIn("running", process.status_text())
        process.set_closure(1.0)
        self.assertTrue(wait_for(lambda: any(n == "MoveGripper" and a[1] == 100 for n, a in controller.calls), 5.0))
        process.set_closure(0.0)
        self.assertTrue(wait_for(lambda: [a[1] for n, a in controller.calls if n == "MoveGripper"][-1] == 0, 5.0))
        self.assertEqual(controller.faults, [], "the controller rejected the argument types")
        activations = [c for c in controller.calls if c[0] == "ActGripper"]
        self.assertEqual(activations, [("ActGripper", (2, 1))])  # once only
        self.assertIn(("MoveGripper", (2, 100, 100, 67, 2000, 1, 0, 0.0, 0, 0)), controller.calls)
        self.assertTrue(process.alive)

    def test_an_unreachable_controller_does_not_hang_the_arm(self):
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]  # nothing listens here
        cfg = process_config(port)
        cfg.fairino.gripper.activation_timeout_s = 1.0
        process = GripperProcess(cfg)
        process.start()
        self.addCleanup(process.stop)
        started = time.monotonic()
        self.assertTrue(process.wait_ready(15.0))  # activation fails, the process carries on
        self.assertLess(time.monotonic() - started, 15.0)


class RobotBackendTests(unittest.TestCase):
    def make_robot(self, controller_port):
        cfg = process_config(controller_port)
        # no servo thread runs in these tests to answer pause requests (tests/test_gripper_pause.py does that)
        cfg.fairino.gripper.pause_arm_for_moves = False
        robot = FairinoRobot(cfg, [-3.0] * 6, [3.0] * 6, sdk=object())
        rpc = FakeRPC()
        rpc.robot_state_pkg = SimpleNamespace(gripper_active=1 << 2, gripper_position=0)
        robot._rpc = rpc
        robot._connected = True
        robot._bind_sdk_calls()
        return robot

    def test_the_backend_starts_the_process_feeds_it_and_waits_for_activation(self):
        controller = StrictControllerServer()
        self.addCleanup(controller.close)
        robot = self.make_robot(controller.port)
        robot._prepare_gripper()
        self.addCleanup(robot._disconnect)
        self.assertIsNotNone(robot._gripper_process)
        self.assertIsNone(robot._gripper_thread)  # no thread of the arm's process touches the gripper
        robot._on_stream_start()  # waits for the activation, then asks for a re-send
        self.assertIn("gripper process running", robot.gripper_status())
        robot.set_gripper(1.0)  # a shared-memory write
        self.assertTrue(wait_for(lambda: any(n == "MoveGripper" and a[1] == 100 for n, a in controller.calls), 5.0))
        self.assertEqual(controller.faults, [])
        robot.set_gripper(0.0)
        robot.resend_gripper()
        self.assertTrue(wait_for(lambda: [a[1] for n, a in controller.calls if n == "MoveGripper"][-1] == 0, 5.0))

    def test_set_gripper_never_blocks_even_if_the_controller_is_silent(self):
        controller = StrictControllerServer()
        controller.close()  # nothing answers any more
        robot = self.make_robot(controller.port)
        robot._prepare_gripper()
        self.addCleanup(robot._disconnect)
        started = time.monotonic()
        for value in (0.2, 0.9, 0.4, 1.0):
            robot.set_gripper(value)
        self.assertLess(time.monotonic() - started, 0.05)

    def test_the_thread_path_is_still_available(self):
        cfg = Config()
        cfg.fairino.gripper.use_process = False
        self.assertFalse(cfg.validate().fairino.gripper.use_process)


if __name__ == "__main__":
    unittest.main()
