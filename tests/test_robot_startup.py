"""Starting servo streaming must preserve the gripper's measured opening."""

import threading
import time
import unittest

import numpy as np

from fr10.config import Config
from fr10.robots.base import Robot, RobotError, RobotState


class CapturingRobot(Robot):
    """Minimal backend exposing every gripper command sent by the real servo thread."""

    def __init__(self, measured_gripper):
        super().__init__(Config().validate(), [-1.0] * 6, [1.0] * 6)
        self.measured_gripper = measured_gripper
        self.commands = []
        self.command_sent = threading.Event()
        self.stream_starts = 0

    def _connect(self):
        pass

    def _disconnect(self):
        pass

    def _read_state(self):
        return RobotState(time.monotonic(), np.zeros(6), np.zeros(6), self.measured_gripper)

    def _send_setpoint(self, q, gripper):
        self.commands.append(gripper)
        self.command_sent.set()

    def _on_stream_start(self):
        self.stream_starts += 1


class RobotStartupTests(unittest.TestCase):
    def make_robot(self, measured_gripper):
        robot = CapturingRobot(measured_gripper)
        robot.connect()
        self.addCleanup(robot.close)
        return robot

    def start_and_capture(self, robot):
        robot.commands.clear()
        robot.command_sent.clear()
        robot.start_streaming()
        self.assertTrue(robot.command_sent.wait(1.0), "servo thread did not send a command")
        robot.stop_streaming()
        return robot.commands[:]

    def test_closed_gripper_never_receives_default_open_command_on_start(self):
        robot = self.make_robot(1.0)
        self.assertEqual(robot.gripper_command, 0.0)

        commands = self.start_and_capture(robot)

        self.assertTrue(commands)
        self.assertTrue(all(command == 1.0 for command in commands), commands)
        self.assertEqual(robot.gripper_command, 1.0)

    def test_restart_resynchronizes_to_new_measured_opening(self):
        robot = self.make_robot(0.2)
        self.assertTrue(all(command == 0.2 for command in self.start_and_capture(robot)))
        robot.set_gripper(0.0)
        robot.measured_gripper = 0.7

        commands = self.start_and_capture(robot)

        self.assertTrue(commands)
        self.assertTrue(all(command == 0.7 for command in commands), commands)

    def test_unknown_feedback_preserves_requested_gripper_command(self):
        robot = self.make_robot(None)
        robot.set_gripper(0.65)

        commands = self.start_and_capture(robot)

        self.assertTrue(commands)
        self.assertTrue(all(command == 0.65 for command in commands), commands)
        self.assertEqual(robot.gripper_command, 0.65)

    def test_nonfinite_gripper_feedback_prevents_streaming(self):
        for gripper in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(gripper=gripper):
                robot = self.make_robot(gripper)
                robot.set_gripper(0.4)

                with self.assertRaisesRegex(RobotError, "gripper feedback is not finite"):
                    robot.start_streaming()

                self.assertFalse(robot.streaming)
                self.assertEqual(robot.stream_starts, 0)
                self.assertEqual(robot.commands, [])
                self.assertEqual(robot.gripper_command, 0.4)


if __name__ == "__main__":
    unittest.main()
