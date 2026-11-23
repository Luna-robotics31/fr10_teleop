"""Exercise the actual TCP/ZeroMQ simulator link without Isaac Sim or hardware."""

from __future__ import annotations

import socket
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from fr10.comm import ArmCommand, ArmState, SimClientLink
from fr10.config import Config
from fr10.robots.sim import SimRobot
from fr10.sim_bridge import SimBridge
from fr10.teleop import Mode, Teleoperator
from fr10.xr_input import OperatorInput


def free_ports():
    """Ask the OS for two different available TCP ports, reserving both together."""
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def config():
    cfg = Config()
    cfg.sim.state_port, cfg.sim.command_port = free_ports()
    cfg.sim.host = "127.0.0.1"
    return cfg


def wait_for(predicate, timeout=2.0):
    """Wait for observable transport progress rather than guessing subscription delay."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError("simulator link did not make progress before the deadline")


class EncodedPayload:
    """A deliberately malformed peer message sent through the normal transport."""

    def __init__(self, payload):
        self.payload = payload

    def encode(self):
        return self.payload


class BridgeLinkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.home = np.asarray(self.cfg.robot.home_rad)
        self.bridge = SimBridge(self.cfg, self.home, view_yaw=0.35, bind_host="127.0.0.1")
        self.addCleanup(self.bridge.close)
        self.client = SimClientLink("127.0.0.1", self.cfg.sim.state_port, self.cfg.sim.command_port)
        self.addCleanup(self.client.close)

    def command(self, seq, q, gripper=None, reset_id=0):
        message = ArmCommand(seq, time.time(), q, gripper, reset_id)

        def exchange():
            self.client.send_command(message)
            received = self.bridge.poll()
            return received if received.fresh and self.bridge._cmd_seq == seq else None

        return wait_for(exchange)

    def test_joint_gripper_and_object_feedback_round_trip(self):
        target = self.home + 0.025
        command = self.command(3, target.tolist(), gripper=0.7)
        np.testing.assert_allclose(command.joint_target, target)
        self.assertAlmostEqual(command.gripper, 0.7)
        self.assertTrue(self.bridge.teleop_connected)
        objects = {"cube_red": [0.55, -0.15, 0.025, 1.0, 0.0, 0.0, 0.0]}

        def feedback():
            self.bridge.publish(1.25, target, np.zeros(6), 0.65, objects)
            return self.client.receive_state(timeout_ms=10)

        state = wait_for(feedback)
        self.assertEqual(state.cmd_seq, 3)
        self.assertEqual(state.objects, objects)
        self.assertAlmostEqual(state.view_yaw, 0.35)
        self.assertAlmostEqual(state.sim_time, 1.25)
        self.assertAlmostEqual(state.gripper, 0.65)
        np.testing.assert_allclose(state.q, target)

    def test_reset_ignores_old_targets_then_accepts_resynchronised_command(self):
        target = self.home + 0.05
        self.command(1, target.tolist(), gripper=0.9)
        reset = self.command(2, target.tolist(), gripper=0.9, reset_id=1)
        self.assertTrue(reset.reset)
        np.testing.assert_allclose(reset.joint_target, self.home)
        self.assertEqual(reset.gripper, 0.0)
        self.assertFalse(self.bridge.poll().reset)

        resumed = self.command(3, target.tolist(), gripper=0.4, reset_id=1)
        self.assertFalse(resumed.reset)
        np.testing.assert_allclose(resumed.joint_target, target)
        self.assertAlmostEqual(resumed.gripper, 0.4)

    def test_command_timeout_holds_the_last_targets(self):
        self.cfg.sim.command_timeout_s = 0.05
        target = self.home + 0.02
        self.command(1, target.tolist(), gripper=0.6)
        self.assertTrue(self.bridge.teleop_connected)

        def inactive():
            self.bridge.poll()  # drain any handshake retry still travelling through TCP
            return not self.bridge.teleop_connected

        wait_for(inactive)
        held = self.bridge.poll()
        self.assertFalse(held.fresh)
        np.testing.assert_allclose(held.joint_target, target)
        self.assertAlmostEqual(held.gripper, 0.6)

    def test_malformed_commands_are_counted_and_valid_stream_recovers(self):
        def malformed():
            self.client.send_command(EncodedPayload(b'{"type":"cmd","v":999}'))
            self.bridge.poll()
            return self.bridge.bad_messages > 0

        wait_for(malformed)
        command = self.command(4, self.home.tolist())
        self.assertTrue(command.fresh)
        np.testing.assert_allclose(command.joint_target, self.home)

    def test_malformed_feedback_is_counted_and_valid_stream_recovers(self):
        def malformed():
            self.bridge._link.publish_state(EncodedPayload(b"not json"))
            self.client.receive_state(timeout_ms=10)
            return self.client.bad_messages > 0

        wait_for(malformed)

        def valid():
            self.bridge.publish(0.1, self.home, np.zeros(6), 0.0)
            return self.client.receive_state(timeout_ms=10)

        state = wait_for(valid)
        self.assertIsInstance(state, ArmState)
        np.testing.assert_allclose(state.q, self.home)


class KinematicSimulator:
    """A small position-following peer; its ZMQ sockets belong to its thread."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, name="test-simulator", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(2.0):
            self.close()
            raise AssertionError("test simulator did not start")
        if self.error is not None:
            raise self.error

    def run(self):
        bridge = None
        try:
            home = np.asarray(self.cfg.robot.home_rad)
            bridge = SimBridge(self.cfg, home, view_yaw=0.35, bind_host="127.0.0.1")
            self.ready.set()
            sim_time = 0.0
            while not self.stop.is_set():
                command = bridge.poll()
                sim_time += 0.005
                bridge.publish(sim_time, command.joint_target, np.zeros(6), command.gripper)
                self.stop.wait(0.005)
        except Exception as exc:
            self.error = exc
            self.ready.set()
        finally:
            if bridge is not None:
                bridge.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise AssertionError("test simulator thread did not stop")
        if self.error is not None:
            raise self.error


class SimRobotLinkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.home = np.asarray(self.cfg.robot.home_rad)
        self.simulator = KinematicSimulator(self.cfg)
        self.addCleanup(self.simulator.close)
        self.simulator.start()
        self.robot = SimRobot(self.cfg, np.full(6, -10.0), np.full(6, 10.0))
        self.addCleanup(self.robot.close)
        self.robot.connect()

    def test_connect_stream_reset_and_shutdown_over_tcp(self):
        self.assertTrue(self.robot.connected)
        self.assertAlmostEqual(self.robot.view_yaw(), 0.35)
        np.testing.assert_allclose(self.robot.get_state().q, self.home)
        self.robot.start_streaming()
        target = self.home + 0.04
        self.robot.set_joint_target(target)
        self.robot.set_gripper(0.6)
        wait_for(lambda: np.allclose(self.robot.get_state().q, target, atol=5e-4))
        wait_for(lambda: abs(self.robot.get_state().gripper - 0.6) < 1e-5)
        self.assertGreater(self.robot.get_state().info["cmd_seq"], 0)

        self.assertTrue(self.robot.reset_scene(timeout=2.0))
        state = self.robot.get_state()
        self.assertEqual(state.info["reset_id"], 1)
        np.testing.assert_allclose(state.q, self.home, atol=1e-5)
        np.testing.assert_allclose(self.robot.commanded_position(), self.home, atol=1e-5)
        self.assertAlmostEqual(state.gripper, 0.0)
        self.assertAlmostEqual(self.robot.gripper_command, 0.0)
        # Let several further servo packets arrive: the old command must not return.
        time.sleep(0.05)
        self.assertAlmostEqual(self.robot.get_state().gripper, 0.0)

        receiver, servo = self.robot._rx_thread, self.robot._thread
        self.robot.close()
        self.assertFalse(self.robot.connected)
        self.assertFalse(self.robot.streaming)
        self.assertFalse(receiver.is_alive())
        self.assertFalse(servo.is_alive())

    def test_reset_is_available_before_streaming(self):
        self.assertFalse(self.robot.streaming)
        self.robot.set_gripper(0.8)
        self.assertTrue(self.robot.reset_scene(timeout=2.0))
        self.assertEqual(self.robot.get_state().info["reset_id"], 1)
        np.testing.assert_allclose(self.robot.commanded_position(), self.home)
        self.assertAlmostEqual(self.robot.gripper_command, 0.0)

    def teleoperator_with_closed_gripper(self, valid=False):
        # These tests never engage arm tracking, so neither IK nor a full robot
        # model is necessary. The robot and simulator still communicate over TCP.
        source = SimpleNamespace(state=OperatorInput(connected=True, valid=valid, grip=1.0))
        source.read = lambda: source.state
        source.wants_images = False
        teleop = Teleoperator(self.cfg, self.robot, SimpleNamespace(home=self.home, fk=lambda q: np.eye(4)), None, source)
        self.addCleanup(teleop.shutdown)
        teleop.post("start")
        teleop.step()
        teleop._gripper = teleop._gripper_desired = 1.0
        teleop._toggle_closed = teleop._prev_grip_high = True
        self.robot.set_gripper(1.0)
        wait_for(lambda: self.robot.get_state().gripper == 1.0)
        return teleop, source

    def test_teleop_reset_does_not_restore_stale_gripper_target_after_input_loss(self):
        for mode in ("analog", "toggle"):
            with self.subTest(mode=mode):
                self.cfg.gripper.mode = mode
                teleop, source = self.teleoperator_with_closed_gripper()
                teleop.post("reset")
                teleop.step()
                self.assertIs(teleop.mode, Mode.READY)
                self.assertEqual(teleop._gripper, 0.0)
                self.assertEqual(teleop._gripper_desired, 0.0)
                self.assertFalse(teleop._toggle_closed)
                self.assertFalse(teleop._prev_grip_high)
                for _ in range(3):
                    teleop.step()
                self.assertEqual(self.robot.gripper_command, 0.0)
                self.assertEqual(self.robot.get_state().gripper, 0.0)
                teleop.shutdown()

    def test_toggle_held_during_reset_requires_a_new_trigger_press(self):
        self.cfg.gripper.mode = "toggle"
        teleop, source = self.teleoperator_with_closed_gripper(valid=True)
        teleop.post("reset")
        teleop.step()
        self.assertFalse(teleop._toggle_closed)
        self.assertTrue(teleop._prev_grip_high)
        self.assertEqual(self.robot.gripper_command, 0.0)
        teleop.step()  # still holding the same trigger press
        self.assertEqual(self.robot.gripper_command, 0.0)
        source.state.grip = 0.0
        teleop.step()
        source.state.grip = 1.0
        teleop.step()
        self.assertTrue(teleop._toggle_closed)
        self.assertGreater(self.robot.gripper_command, 0.0)


if __name__ == "__main__":
    unittest.main()
