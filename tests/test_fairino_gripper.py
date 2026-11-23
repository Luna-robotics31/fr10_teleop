"""FAIRINO gripper open/close: the controller must receive exactly the calls of the standalone scripts."""

import unittest

from fr10.config import Config
from fr10.robots.base import RobotError
from fr10.robots.fairino import FairinoRobot


class FakeRPC:
    """Records gripper calls; ``MoveGripper`` has the full 10-argument vendor signature."""

    def __init__(self, move_code=0, act_code=0):
        self.calls = []
        self.move_code = move_code
        self.act_code = act_code

    def ActGripper(self, index, action):
        self.calls.append(("ActGripper", index, action))
        return self.act_code

    def MoveGripper(self, index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque):
        self.calls.append(("MoveGripper", index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque))
        return self.move_code

    def ServoJ(self, joint_pos, axisPos, acc=0.0, vel=0.0, cmdT=0.008, filterT=0.0, gain=0.0, id=0, cmdType=0):
        return 0  # the signature of the real SDK, UDP pass-through (cmdType) included


class ShortRPC(FakeRPC):
    """Older SDK: MoveGripper without the rotation arguments."""

    def MoveGripper(self, index, pos, vel, force, maxtime, block):
        self.calls.append(("MoveGripper", index, pos, vel, force, maxtime, block))
        return self.move_code


def make_robot(rpc, cfg=None):
    cfg = cfg or Config().validate()
    robot = FairinoRobot(cfg, [-3.0] * 6, [3.0] * 6, sdk=object())
    robot._rpc = rpc
    robot._connected = True
    robot._bind_sdk_calls()
    return robot


class GripperTests(unittest.TestCase):
    def test_open_matches_the_standalone_script(self):
        rpc = FakeRPC()
        robot = make_robot(rpc)
        robot.open_gripper()
        self.assertEqual(rpc.calls, [("ActGripper", 2, 1), ("MoveGripper", 2, 0, 100, 67, 2000, 0, 0, 0, 0, 0)])
        self.assertEqual(robot.gripper_command, 0.0)

    def test_close_matches_the_standalone_script(self):
        rpc = FakeRPC()
        robot = make_robot(rpc)
        robot.close_gripper()
        self.assertEqual(rpc.calls, [("ActGripper", 2, 1), ("MoveGripper", 2, 100, 100, 67, 2000, 0, 0, 0, 0, 0)])
        self.assertEqual(robot.gripper_command, 1.0)

    def test_without_waiting_or_activation(self):
        rpc = FakeRPC()
        robot = make_robot(rpc)
        robot.close_gripper(wait=False, activate=False)
        self.assertEqual(rpc.calls, [("MoveGripper", 2, 100, 100, 67, 2000, 1, 0, 0, 0, 0)])

    def test_never_blocks_while_arm_setpoints_are_streaming(self):
        rpc = FakeRPC()
        robot = make_robot(rpc)
        robot._streaming.set()  # a blocking call would stall ServoJ for up to max_time_ms
        robot.open_gripper(wait=True)
        self.assertEqual(rpc.calls[-1][6], 1)  # block = 1 (do not wait)

    def test_configured_values_are_used(self):
        cfg = Config()
        cfg.fairino.gripper.index = 1
        cfg.fairino.gripper.speed_percent = 40
        cfg.fairino.gripper.force_percent = 20
        cfg.fairino.gripper.max_time_ms = 1500
        cfg.fairino.gripper.closed_percent = 80.0
        rpc = FakeRPC()
        robot = make_robot(rpc, cfg.validate())
        robot.close_gripper()
        self.assertEqual(rpc.calls, [("ActGripper", 1, 1), ("MoveGripper", 1, 80, 40, 20, 1500, 0, 0, 0, 0, 0)])

    def test_older_sdk_signature(self):
        rpc = ShortRPC()
        robot = make_robot(rpc)
        robot.open_gripper()
        self.assertEqual(rpc.calls[-1], ("MoveGripper", 2, 0, 100, 67, 2000, 0))

    def test_controller_errors_are_reported(self):
        with self.assertRaisesRegex(RobotError, "MoveGripper.*error code 7"):
            make_robot(FakeRPC(move_code=7)).close_gripper()
        robot = make_robot(FakeRPC(act_code=3))
        with self.assertRaisesRegex(RobotError, "ActGripper.*error code 3"):
            robot.open_gripper()
        self.assertEqual(robot._rpc.calls, [("ActGripper", 2, 1)])  # no move after a failed activation

    def test_refuses_when_disabled_or_disconnected(self):
        cfg = Config()
        cfg.fairino.gripper.enabled = False
        with self.assertRaisesRegex(RobotError, "disabled"):
            make_robot(FakeRPC(), cfg.validate()).open_gripper()
        robot = make_robot(FakeRPC())
        robot._connected = False
        with self.assertRaisesRegex(RobotError, "not connected"):
            robot.open_gripper()


class HaltTests(unittest.TestCase):
    def test_halt_stops_streaming_and_asks_the_controller_to_stop(self):
        robot = make_robot(FakeRPC())
        robot._streaming.set()
        sent = []
        robot._rpc_call = lambda method, *args: sent.append(method) or 0
        robot.halt()
        self.assertFalse(robot.streaming)
        self.assertEqual(sent, ["StopMotion"])

    def test_halt_survives_a_dead_controller_link(self):
        robot = make_robot(FakeRPC())

        def unreachable(method, *args):
            raise RobotError("no answer")

        robot._rpc_call = unreachable
        robot.halt()  # must not raise
        self.assertFalse(robot.streaming)


class ServoTransportTests(unittest.TestCase):
    """ServoJ goes over the SDK's UDP pass-through by default, so gripper calls cannot stall the arm."""

    class UdpRPC(FakeRPC):
        def __init__(self):
            super().__init__()
            self.servo = []

        def ServoJ(self, joint_pos, axisPos, acc=0.0, vel=0.0, cmdT=0.008, filterT=0.0, gain=0.0, id=0, cmdType=0):
            self.servo.append(("ServoJ", list(joint_pos), cmdT, cmdType))
            return 0

        def ServoMoveStart(self, cmdType=0):
            self.servo.append(("ServoMoveStart", cmdType))
            return 0

        def ServoMoveEnd(self, cmdType=0):
            self.servo.append(("ServoMoveEnd", cmdType))
            return 0

    class OldRPC(FakeRPC):
        def ServoJ(self, joint_pos, axisPos, acc=0.0, vel=0.0, cmdT=0.008, filterT=0.0, gain=0.0, id=0):
            return 0

    def test_udp_is_the_default_and_all_servo_calls_use_it(self):
        cfg = Config().validate()
        self.assertEqual(cfg.fairino.servo_transport, "udp")
        rpc = self.UdpRPC()
        robot = make_robot(rpc, cfg)
        robot._servo_j([1.0] * 6)
        robot._on_stream_start()
        robot._on_stream_stop()
        self.assertEqual(rpc.servo[0][0:2], ("ServoJ", [1.0] * 6))
        self.assertEqual(rpc.servo[0][3], 1)  # cmdType = 1: UDP
        self.assertAlmostEqual(rpc.servo[0][2], 0.008)
        self.assertEqual(rpc.servo[1:], [("ServoMoveStart", 1), ("ServoMoveEnd", 1)])

    def test_xmlrpc_can_still_be_chosen(self):
        cfg = Config()
        cfg.fairino.servo_transport = "xmlrpc"
        rpc = self.UdpRPC()
        robot = make_robot(rpc, cfg.validate())
        robot._servo_j([0.5] * 6)
        robot._on_stream_start()
        self.assertEqual(rpc.servo[0][3], 0)
        self.assertEqual(rpc.servo[1], ("ServoMoveStart", 0))

    def test_an_sdk_without_udp_gets_a_clear_error_and_a_hint(self):
        from fr10.robots.base import RobotError

        with self.assertRaisesRegex(RobotError, "no UDP ServoJ.*xmlrpc"):
            make_robot(self.OldRPC())
        robot = make_robot(self.UdpRPC())
        self.assertIn("servo_transport=xmlrpc", robot.stall_hint())
        cfg = Config()
        cfg.fairino.servo_transport = "xmlrpc"
        self.assertEqual(make_robot(self.UdpRPC(), cfg.validate()).stall_hint(), "")


if __name__ == "__main__":
    unittest.main()
