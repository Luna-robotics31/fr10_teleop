"""Pinch tool next to the teleoperation: shared hand data, plain controller connection, proportional pinch."""

import socket
import time
import unittest

import numpy as np

from fr10.config import Config
from fr10.hand_bus import HandBusPublisher, HandBusSubscriber
from fr10.pinch_gripper import PlainRpc, ProportionalPinchController, ScriptGripper
from fr10.robots.base import RobotError
from tests.test_command_lead import StrictControllerServer, wait_for


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeSource:
    """Stands in for the headset input: ``hand_snapshot(side)``."""

    def __init__(self):
        self.data = {"left": (None, None), "right": (None, None)}

    def hand_snapshot(self, side):
        return self.data[side]


def hand(gap=0.02):
    points = np.full((25, 3), 0.05)
    points[9] = [0.2, 0.0, 0.0]
    points[4] = [0.2, gap, 0.0]
    return points


class HandBusTests(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.publisher = HandBusPublisher(self.port, rate_hz=1000.0)
        self.subscriber = HandBusSubscriber("127.0.0.1", self.port)
        self.addCleanup(self.publisher.close)
        self.addCleanup(self.subscriber.close)
        self.source = FakeSource()

    def pump(self, until, seconds=3.0):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.publisher.update(self.source)
            if until():
                return True
            time.sleep(0.01)
        return False

    def test_hand_data_arrives_with_its_age(self):
        self.source.data["right"] = (hand(0.031), 0.05)
        self.assertTrue(self.pump(lambda: self.subscriber.hand_snapshot("right")[0] is not None))
        positions, age = self.subscriber.hand_snapshot("right")
        self.assertEqual(positions.shape, (25, 3))
        self.assertAlmostEqual(float(np.linalg.norm(positions[4] - positions[9])), 0.031, places=4)
        self.assertGreaterEqual(age, 0.05)  # the age reported by the headset side, plus the transit time
        self.assertLess(age, 0.5)
        self.assertTrue(self.subscriber.connected)

    def test_an_untracked_hand_and_a_silent_publisher(self):
        self.source.data["left"] = (None, None)
        self.assertTrue(self.pump(lambda: self.subscriber.connected))
        self.assertEqual(self.subscriber.hand_snapshot("left"), (None, None))
        end = time.monotonic() + 1.3  # the teleoperation program stopped publishing; the tool keeps reading
        while time.monotonic() < end:
            self.subscriber.hand_snapshot("right")
            time.sleep(0.03)
        self.assertFalse(self.subscriber.connected)
        self.assertEqual(self.subscriber.hand_snapshot("right"), (None, None))

    def test_bad_messages_are_ignored(self):
        import zmq

        context = zmq.Context()
        socket_ = context.socket(zmq.PUB)
        port = free_port()
        socket_.bind(f"tcp://127.0.0.1:{port}")
        subscriber = HandBusSubscriber("127.0.0.1", port)
        try:
            end = time.monotonic() + 3.0
            while subscriber.bad_messages == 0 and time.monotonic() < end:
                socket_.send(b"not json")
                time.sleep(0.02)
                subscriber.hand_snapshot("right")  # reading the socket is what counts the bad messages
            self.assertGreater(subscriber.bad_messages, 0)
            self.assertEqual(subscriber.hand_snapshot("right"), (None, None))
        finally:
            subscriber.close()
            socket_.close(0)
            context.term()

    def test_publisher_needs_a_free_port_and_a_source_with_hands(self):
        with self.assertRaisesRegex(RuntimeError, "cannot bind"):
            HandBusPublisher(self.port)  # already in use
        self.assertFalse(self.publisher.update(object()))  # e.g. the scripted input has no hands


class PlainRpcTests(unittest.TestCase):
    def test_the_controller_accepts_the_plain_calls(self):
        """A type-strict server (as the controller): rotNum must be a double, the rest integers."""
        controller = StrictControllerServer()
        self.addCleanup(controller.close)
        rpc = PlainRpc("127.0.0.1", timeout=2.0)
        rpc.PORT = controller.port
        rpc._proxy = rpc._slow_proxy = __import__("xmlrpc.client").client.ServerProxy(
            f"http://127.0.0.1:{controller.port}"
        )
        cfg = Config().validate()
        gripper = ScriptGripper(cfg, rpc)
        gripper.close_gripper()
        gripper.open_gripper()
        self.assertEqual(controller.faults, [])
        self.assertEqual(
            controller.calls,
            [
                ("ActGripper", (2, 1)),
                ("MoveGripper", (2, 100, 100, 67, 2000, 0, 0, 0.0, 0, 0)),
                ("ActGripper", (2, 1)),
                ("MoveGripper", (2, 0, 100, 67, 2000, 0, 0, 0.0, 0, 0)),
            ],
        )

    def test_a_missing_answer_is_an_error_not_a_hang(self):
        port = free_port()  # nothing listens
        gripper = ScriptGripper(Config().validate(), PlainRpc("127.0.0.1", timeout=0.5))
        gripper._rpc.PORT = port
        gripper._rpc._proxy = gripper._rpc._slow_proxy = __import__("xmlrpc.client").client.ServerProxy(
            f"http://127.0.0.1:{port}"
        )
        started = time.monotonic()
        with self.assertRaisesRegex(RobotError, "no answer"):
            gripper.close_gripper()
        self.assertLess(time.monotonic() - started, 3.0)


class RecordingGripper:
    def __init__(self):
        self.moves, self.fail = [], 0

    def move_percent(self, percent, activate=None, block=None, log=True):
        if self.fail:
            self.fail -= 1
            raise RobotError("refused")
        self.moves.append((int(round(percent)), activate, block))


def feed(controller, distances, start=0.0, dt=0.02):
    return [controller.update(d, start + i * dt) for i, d in enumerate(distances)]


class ProportionalPinchTests(unittest.TestCase):
    def test_half_pinch_stops_halfway_and_full_pinch_closes(self):
        gripper = RecordingGripper()
        controller = ProportionalPinchController(gripper)
        feed(controller, [0.09] * 10)  # open hand
        feed(controller, list(np.linspace(0.09, 0.05, 30)), start=1.0)  # closing, then stopping at 50 mm
        feed(controller, [0.05] * 60, start=2.0)  # held: nothing more may be sent
        halfway = list(gripper.moves)
        self.assertAlmostEqual(halfway[-1][0], 50, delta=3)
        self.assertEqual(sorted(set(m[0] for m in halfway)), sorted(m[0] for m in halfway))  # monotonic, no repeats
        feed(controller, [0.015] * 40, start=4.0)  # full pinch
        self.assertEqual(gripper.moves[-1][0], 100)
        feed(controller, [0.11] * 40, start=6.0)  # open hand
        self.assertEqual(gripper.moves[-1][0], 0)

    def test_the_controller_leaves_activation_and_waiting_to_the_gripper(self):
        gripper = RecordingGripper()
        controller = ProportionalPinchController(gripper)
        feed(controller, [0.015] * 20)
        feed(controller, [0.11] * 40, start=1.0)
        self.assertEqual(gripper.moves[0][1:], (None, None))  # never decides activation itself,
        self.assertEqual(gripper.moves[-1][1:], (None, None))  # not even after a failure (activating is slow)

    def test_noise_a_lost_hand_and_refusals(self):
        gripper = RecordingGripper()
        controller = ProportionalPinchController(gripper)
        rng = np.random.default_rng(0)
        feed(controller, list(0.05 + rng.uniform(-0.002, 0.002, 100)))
        count = len(gripper.moves)
        feed(controller, list(0.05 + rng.uniform(-0.002, 0.002, 100)), start=2.0)
        self.assertEqual(len(gripper.moves), count)  # jitter around a held pinch sends nothing
        feed(controller, [None] * 50, start=4.0)  # hand lost: gripper stays
        self.assertEqual(len(gripper.moves), count)
        gripper.fail = 2
        feed(controller, [0.015] * 120, start=6.0, dt=0.02)  # two refusals, then it goes through
        self.assertEqual(gripper.moves[-1][0], 100)
        self.assertEqual(controller.failures, 0)

    def test_few_commands_while_the_fingers_move_and_one_exact_final_command(self):
        """Gripper commands share the controller with the arm: a full pinch may cost only a few of them."""
        gripper = RecordingGripper()
        controller = ProportionalPinchController(gripper)
        feed(controller, [0.10] * 25)  # open hand
        before = len(gripper.moves)
        times = []
        for i, d in enumerate(np.linspace(0.10, 0.052, 100)):  # fingers close over 2 s and stop at 52 mm
            if controller.update(float(d), 1.0 + i * 0.02) is not None:
                times.append(1.0 + i * 0.02)
        feed(controller, [0.052] * 100, start=3.0)  # held still
        moves = gripper.moves[before:]
        self.assertLessEqual(len(moves), 7)  # a few spaced commands, not one every few milliseconds
        gaps = np.diff(times)
        self.assertTrue(np.all(gaps >= 0.4 - 1e-9), gaps)  # at most 2.5 commands a second while moving
        self.assertAlmostEqual(moves[-1][0], 47, delta=3)  # the settle command lands on the exact position
        self.assertEqual(len(gripper.moves), before + len(moves))
        count = len(gripper.moves)
        feed(controller, [0.052] * 200, start=6.0)  # holding forever costs nothing
        self.assertEqual(len(gripper.moves), count)

    def test_invalid_range(self):
        with self.assertRaises(ValueError):
            ProportionalPinchController(RecordingGripper(), open_m=0.02, closed_m=0.05)


if __name__ == "__main__":
    unittest.main()
