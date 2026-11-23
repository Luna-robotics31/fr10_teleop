"""Gripper moves during teleoperation: the arm pauses for each one, because the FR10 controller needs it.

Measured on the real controller (V3.9.4): while servo set-points stream, the controller takes one
gripper move; every later MoveGripper waits until the stream has been quiet for ~1.2 s. The fake
controller below follows the same rule (with a shorter quiet time) so the whole chain can be tested:
teleoperation -> gripper process -> pause handshake -> servo thread -> controller.
"""

import math
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from fr10.config import Config, ConfigError, load_config
from fr10.gripper_process import ARM_PAUSED_FOR, COMMANDS, FAILURES, STATUS_LEN, GripperProcess, gripper_main
from fr10.kinematics import RobotModel
from fr10.pinch_gripper import ProportionalPinchController
from fr10.robots.base import RobotError
from fr10.robots.fairino import FairinoRobot
from fr10.robots.mock import MockRobot
from fr10.trajectory import JointSetpointFilter
from tests.test_command_lead import StrictControllerServer, wait_for
from tests.test_fairino_gripper import FakeRPC


# ---------------------------------------------------------------------------------------------- braking
class StoppingPointTests(unittest.TestCase):
    def test_braking_to_the_stopping_point_is_smooth_and_never_turns_back(self):
        a_max, dt = 8.0, 0.008
        flt = JointSetpointFilter([2.0] * 3, [a_max] * 3, [-6.0] * 3, [6.0] * 3)
        flt.reset(np.zeros(3))
        for _ in range(60):
            flt.step([5.0, -5.0, 0.0], dt)  # moving: two joints fast, one still
        v0 = flt.velocity
        self.assertGreater(abs(v0[0]), 0.4)
        stop = flt.stopping_point()
        positions, velocities = [flt.position], [v0]
        for _ in range(400):
            positions.append(flt.step(stop, dt))
            velocities.append(flt.velocity)
        velocities = np.array(velocities)
        accel = np.abs(np.diff(velocities, axis=0)) / dt
        self.assertLessEqual(float(accel.max()), a_max + 1e-6)  # never harder than the acceleration limit
        for j in range(3):  # each joint slows down monotonically and never reverses
            self.assertTrue(np.all(np.sign(velocities[:, j]) * np.sign(v0[j]) >= -1e-9))
            self.assertTrue(np.all(np.diff(np.abs(velocities[:, j])) <= 1e-9))
        np.testing.assert_allclose(positions[-1], stop, atol=1e-4)
        self.assertLess(float(np.max(np.abs(flt.velocity))), 1e-3)


# ------------------------------------------------------------------------------------ servo thread side
class PausableMock(MockRobot):
    """A mock arm whose backend can ask for pauses, recording every set-point it is sent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.want_pause = threading.Event()
        self.notifications = []
        self.sent = []  # (time, q) of every set-point that was actually sent

    def _pause_requested(self):
        return self.want_pause.is_set()

    def _set_paused(self, paused):
        self.notifications.append((time.monotonic(), paused))

    def _send_setpoint(self, q, gripper):
        if q is not None:
            self.sent.append((time.monotonic(), np.array(q)))
        super()._send_setpoint(q, gripper)


class StreamPauseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel(Config().validate().robot)

    def robot(self, cfg=None):
        cfg = cfg or Config().validate()
        robot = PausableMock(cfg, self.model.lower, self.model.upper, lag_s=0.02)
        robot.connect()
        self.addCleanup(robot.close)
        return robot

    def test_the_arm_brakes_smoothly_stays_silent_and_then_follows_again(self):
        robot = self.robot()
        robot.start_streaming()
        start = robot.get_state().q.copy()
        goal = start + math.radians(25.0)
        robot.set_joint_target(goal)
        time.sleep(0.25)  # moving at speed
        robot.want_pause.set()
        self.assertTrue(wait_for(lambda: [n for _, n in robot.notifications] == [True], 3.0))
        paused_at = robot.notifications[0][0]
        held = robot.commanded_position()
        time.sleep(0.5)
        self.assertEqual([s for s in robot.sent if s[0] > paused_at + 1e-3], [])  # nothing sent while paused
        np.testing.assert_array_equal(robot.commanded_position(), held)
        self.assertTrue(robot.paused)
        robot.want_pause.clear()
        self.assertTrue(wait_for(lambda: len(robot.notifications) == 2, 1.0))
        self.assertEqual(robot.notifications[1][1], False)
        self.assertTrue(wait_for(lambda: np.max(np.abs(robot.get_state().q - goal)) < math.radians(0.5), 10.0))
        self.assertFalse(robot.paused)
        self.assertEqual(robot.pauses, 1)

        # the whole stream of set-points stays within the acceleration limits (braking and restarting)
        t = np.array([s[0] for s in robot.sent])
        q = np.array([s[1] for s in robot.sent])
        dt = robot.servo_period
        moving = np.diff(t) < 5 * dt  # the pause itself is a gap, not a step
        v = np.diff(q, axis=0) / dt
        accel = np.abs(np.diff(v, axis=0)) / dt
        accel = accel[moving[1:] & moving[:-1]]
        a_max = np.array(robot.cfg.robot.max_acceleration_rad_s2) * robot.cfg.control.speed_scale
        self.assertTrue(np.all(accel <= a_max * 1.01 + 1e-6), f"max {np.max(accel / a_max):.2f} x the limit")
        # and resumes from rest at the held position: no jump across the pause
        gap = int(np.argmax(~moving))
        np.testing.assert_allclose(q[gap + 1], q[gap], atol=float(np.max(a_max)) * dt * dt * 2)

    def test_a_request_that_is_never_withdrawn_does_not_freeze_the_arm(self):
        robot = self.robot()
        robot.pause_timeout_s = 0.4
        robot.start_streaming()
        goal = robot.get_state().q + math.radians(10.0)
        robot.want_pause.set()
        self.assertTrue(wait_for(lambda: [n for _, n in robot.notifications] == [True], 2.0))
        robot.set_joint_target(goal)
        self.assertTrue(wait_for(lambda: len(robot.notifications) == 2, 2.0))  # timed out: running again
        self.assertTrue(wait_for(lambda: np.max(np.abs(robot.get_state().q - goal)) < math.radians(0.5), 10.0))
        self.assertEqual(robot.pauses, 1)  # the stale request is ignored ...
        robot.want_pause.clear()
        time.sleep(0.05)
        robot.want_pause.set()  # ... until it is withdrawn; a new one pauses again
        self.assertTrue(wait_for(lambda: robot.pauses == 2 and [n for _, n in robot.notifications][-1:] == [True], 2.0))

    def test_stopping_while_paused_tells_the_backend(self):
        robot = self.robot()
        robot.start_streaming()
        robot.want_pause.set()
        self.assertTrue(wait_for(lambda: [n for _, n in robot.notifications][-1:] == [True], 2.0))
        robot.stop_streaming()
        self.assertEqual(robot.notifications[-1][1], False)
        self.assertFalse(robot.paused)

    def test_the_stream_stays_silent_until_the_backend_says_it_may_go_on(self):
        robot = self.robot()
        may_go_on = threading.Event()
        robot._pause_can_end = lambda since_withdrawn: may_go_on.is_set()  # e.g. "the gripper move is done"
        robot.start_streaming()
        robot.want_pause.set()
        self.assertTrue(wait_for(lambda: [n for _, n in robot.notifications] == [True], 2.0))
        robot.want_pause.clear()  # the request is withdrawn ...
        time.sleep(0.1)
        sent = len(robot.sent)
        time.sleep(0.3)
        self.assertEqual(len(robot.sent), sent)  # ... but nothing is sent until the backend allows it
        self.assertTrue(robot.paused)
        may_go_on.set()
        self.assertTrue(wait_for(lambda: [n for _, n in robot.notifications] == [True, False], 1.0))
        self.assertTrue(wait_for(lambda: len(robot.sent) > sent, 1.0))
        self.assertFalse(robot.paused)

    def test_latency_guard_sends_nothing_while_the_robot_queue_is_long(self):
        robot = self.robot()
        robot.max_backlog_points = 5
        backlog = [0]
        robot._command_backlog = lambda: backlog[0]
        robot.start_streaming()
        goal = robot.get_state().q + math.radians(10.0)
        robot.set_joint_target(goal)
        time.sleep(0.2)
        backlog[0] = 50  # something stalled the robot's own queue
        time.sleep(0.05)
        sent, held = len(robot.sent), robot.commanded_position()
        time.sleep(0.3)
        self.assertLessEqual(len(robot.sent), sent + 1)  # (one tick may have been under way)
        self.assertLess(float(np.max(np.abs(robot.commanded_position() - held))), math.radians(0.5))
        self.assertGreater(robot.lag_hold_time, 0.2)  # a queue that never drains ends in the stall fault
        self.assertGreater(robot.drain_ticks, 20)
        backlog[0] = 0
        self.assertTrue(wait_for(lambda: np.max(np.abs(robot.get_state().q - goal)) < math.radians(0.5), 10.0))
        self.assertEqual(robot.lag_hold_time, 0.0)

    def test_a_small_persistent_backlog_is_trimmed_and_jitter_is_ignored(self):
        robot = self.robot()
        robot.max_backlog_points = 10
        backlog = [1]
        robot._command_backlog = lambda: backlog[0]
        robot.start_streaming()
        time.sleep(0.5)
        self.assertEqual(robot.drain_ticks, 0)  # 0-1 queued is normal: never trimmed
        backlog[0] = 2  # a persistent backlog of two set-points (16 ms of extra lag)
        time.sleep(1.0)
        trimmed = robot.drain_ticks
        self.assertGreaterEqual(trimmed, 5)  # one tick skipped at most every 11 ticks ...
        self.assertLessEqual(trimmed, 14)  # ... never in a burst
        self.assertEqual(robot.lag_hold_time, 0.0)  # trimming is not a stall

    def test_a_long_queue_is_drained_down_to_one_before_streaming_again(self):
        robot = self.robot()
        robot.max_backlog_points = 10
        backlog = [0]
        robot._command_backlog = lambda: backlog[0]
        robot.start_streaming()
        time.sleep(0.1)
        backlog[0] = 12
        time.sleep(0.1)
        sent = len(robot.sent)
        backlog[0] = 5  # below the limit, but still draining (hysteresis)
        time.sleep(0.2)
        self.assertLessEqual(len(robot.sent), sent + 1)
        backlog[0] = 1
        self.assertTrue(wait_for(lambda: len(robot.sent) > sent + 5, 1.0))

    def test_backends_without_pauses_are_unchanged(self):
        cfg = Config().validate()
        robot = MockRobot(cfg, self.model.lower, self.model.upper, lag_s=0.02)
        robot.connect()
        self.addCleanup(robot.close)
        robot.start_streaming()
        goal = robot.get_state().q + math.radians(5.0)
        robot.set_joint_target(goal)
        self.assertTrue(wait_for(lambda: np.max(np.abs(robot.get_state().q - goal)) < math.radians(0.5), 5.0))
        self.assertEqual(robot.pauses, 0)


# --------------------------------------------------------------------------------- gripper process side
class FakeArm:
    """Answers pause requests the way the servo thread does (after a short braking time)."""

    def __init__(self, brake_s=0.05, responsive=True):
        self.streaming = SimpleNamespace(value=1)
        self.pause_request, self.arm_paused = threading.Event(), threading.Event()
        self.brake_s, self.responsive = brake_s, responsive
        self.pauses = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if self.pause_request.is_set() and not self.arm_paused.is_set() and self.responsive:
                time.sleep(self.brake_s)
                self.pauses += 1
                self.arm_paused.set()
            elif not self.pause_request.is_set() and self.arm_paused.is_set():
                self.arm_paused.clear()
            time.sleep(0.002)

    def close(self):
        self._stop.set()
        self._thread.join(1.0)


class PauseCheckingGripper:
    """Records each move and whether the arm was paused when it was sent."""

    def __init__(self, arm):
        self.arm = arm
        self.moves = []
        self.activate_each_move, self.block = False, 1

    def activate(self):
        pass

    def move_percent(self, percent, **kwargs):
        self.moves.append((int(round(percent)), self.arm.arm_paused.is_set()))

    def close_connection(self):
        pass


class ProcessLoopTests(unittest.TestCase):
    def run_loop(self, arm, gripper, closure, cfg=None):
        cfg = cfg or Config().validate()
        cfg.fairino.gripper.pause_cooldown_s = 0.1
        target = SimpleNamespace(value=closure)
        status = [0.0] * STATUS_LEN
        stop, ready = threading.Event(), threading.Event()
        thread = threading.Thread(
            target=gripper_main,
            args=(cfg, target, status, threading.Event(), stop, ready, lambda c: gripper),
            kwargs=dict(streaming=arm.streaming, pause_request=arm.pause_request, arm_paused=arm.arm_paused),
            daemon=True,
        )
        thread.start()
        self.assertTrue(ready.wait(3.0))

        def finish():
            stop.set()
            thread.join(3.0)
            arm.close()

        self.addCleanup(finish)
        return target, status

    def test_every_move_during_streaming_is_sent_while_the_arm_is_paused(self):
        arm = FakeArm()
        gripper = PauseCheckingGripper(arm)
        target, status = self.run_loop(arm, gripper, closure=0.0)
        self.assertTrue(wait_for(lambda: gripper.moves, 3.0))
        target.value = 1.0
        self.assertTrue(wait_for(lambda: gripper.moves[-1][0] == 100, 3.0))
        target.value = 0.0
        self.assertTrue(wait_for(lambda: gripper.moves[-1][0] == 0, 3.0))
        self.assertTrue(all(paused for _, paused in gripper.moves), gripper.moves)
        self.assertTrue(wait_for(lambda: not arm.pause_request.is_set(), 1.0))  # the arm is released
        self.assertEqual(status[ARM_PAUSED_FOR], 0.0)
        self.assertEqual(arm.pauses, len(gripper.moves))

    def test_no_pause_when_the_arm_is_not_streaming(self):
        arm = FakeArm()
        arm.streaming.value = 0
        gripper = PauseCheckingGripper(arm)
        target, _ = self.run_loop(arm, gripper, closure=1.0)
        self.assertTrue(wait_for(lambda: gripper.moves, 3.0))
        self.assertEqual(gripper.moves, [(100, False)])
        self.assertEqual(arm.pauses, 0)

    def test_an_arm_that_does_not_pause_gets_no_move_and_is_released(self):
        arm = FakeArm(responsive=False)
        gripper = PauseCheckingGripper(arm)
        import fr10.gripper_process as gp

        original, gp.ARM_BRAKE_TIMEOUT_S = gp.ARM_BRAKE_TIMEOUT_S, 0.2
        self.addCleanup(setattr, gp, "ARM_BRAKE_TIMEOUT_S", original)
        target, status = self.run_loop(arm, gripper, closure=1.0)
        self.assertTrue(wait_for(lambda: status[FAILURES] >= 1, 3.0))
        self.assertEqual(gripper.moves, [])  # never sent into a streaming controller
        self.assertTrue(wait_for(lambda: not arm.pause_request.is_set(), 1.0))

    def test_moving_fingers_cost_one_pause_per_gesture(self):
        arm = FakeArm()
        gripper = PauseCheckingGripper(arm)
        target, _ = self.run_loop(arm, gripper, closure=0.0)
        self.assertTrue(wait_for(lambda: gripper.moves, 3.0))
        before = len(gripper.moves)
        for value in np.linspace(0.0, 0.6, 15):  # the fingers close slowly over 0.75 s and stop at 60 %
            target.value = float(value)
            time.sleep(0.05)
        self.assertTrue(wait_for(lambda: len(gripper.moves) > before, 3.0))
        time.sleep(0.6)
        sent = gripper.moves[before:]
        self.assertEqual(len(sent), 1, sent)
        self.assertAlmostEqual(sent[0][0], 60, delta=2)


class SettledOnlyTests(unittest.TestCase):
    def test_jitter_within_the_band_still_counts_as_settled(self):
        moves = []
        gripper = SimpleNamespace(move_percent=lambda p, **k: moves.append(p))
        follower = ProportionalPinchController(gripper, min_interval_s=0.0, settled_only=True, still_band=4.0)
        rng = np.random.default_rng(1)
        now = 0.0
        for _ in range(40):  # half closed, with +-1.5 % of tracking jitter
            follower.update_closure(0.5 + rng.uniform(-0.015, 0.015), now)
            now += 0.05
        self.assertEqual(len(moves), 1)
        self.assertAlmostEqual(moves[0], 50, delta=2)

    def test_fully_closed_is_sent_at_once(self):
        moves = []
        gripper = SimpleNamespace(move_percent=lambda p, **k: moves.append(p))
        follower = ProportionalPinchController(gripper, min_interval_s=0.0, settled_only=True)
        follower.update_closure(1.0, 0.0)
        self.assertEqual(moves, [100])


class ConfigTests(unittest.TestCase):
    def test_defaults_and_checks(self):
        gc = Config().validate().fairino.gripper
        self.assertTrue(gc.pause_arm_for_moves)
        self.assertTrue(gc.use_process)
        with self.assertRaises(ConfigError):
            load_config(overrides={"fairino.gripper.pause_timeout_s": 3.0})  # shorter than a gripper call may take
        with self.assertRaises(ConfigError):
            load_config(overrides={"fairino.gripper.pause_cooldown_s": -1.0})


# ------------------------------------------------------------------------------------- the whole chain
class QueueController(StrictControllerServer):
    """The FR10 controller as measured: ServoJ set-points and MoveGripper share one motion queue.

    Set-points run one per servo period. A gripper move is taken when the queue is empty and then holds
    the queue for its duration (the arm stands still; set-points sent meanwhile pile up and, arriving as
    fast as they run, never drain). The state packet reports mc_queue_len and gripper_motiondone.
    """

    PERIOD = 0.008
    GRIPPER_S = 0.4  # ~1.2 s on the real controller for a full stroke

    def __init__(self):
        self.queue = 0
        self.max_queue_after_start = 0
        self.gripper_until = 0.0
        self.servo_count = 0
        self.queued_behind_gripper = 0
        self.lock = threading.Lock()
        super().__init__()
        self.server.funcs["MoveGripper"] = self._move_gripper
        self._run = True
        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()

    def close(self):
        self._run = False
        super().close()

    def gripper_moving(self):
        return time.monotonic() < self.gripper_until

    def _consume(self):
        next_tick = time.monotonic()
        while self._run:
            with self.lock:
                if not self.gripper_moving() and self.queue > 0:
                    self.queue -= 1
            next_tick += self.PERIOD
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def servo(self):
        with self.lock:
            self.queue += 1
            self.servo_count += 1
            if self.gripper_moving():
                self.queued_behind_gripper += 1
            self.max_queue_after_start = max(self.max_queue_after_start, self.queue)

    def _move_gripper(self, *args):
        arrived = time.monotonic()
        while True:
            with self.lock:
                if self.queue == 0 and not self.gripper_moving():
                    self.gripper_until = time.monotonic() + self.GRIPPER_S
                    break
            if time.monotonic() - arrived > 10.0:
                raise RuntimeError("the queue never emptied")
            time.sleep(0.002)
        self.calls.append(("MoveGripper", tuple(args)))
        return 0


class StreamingRPC(FakeRPC):
    """SDK stand-in: ServoJ reaches the fake controller and the joint feedback follows it."""

    def __init__(self, controller, q0_deg):
        super().__init__()
        self.controller = controller
        self.q_deg = list(q0_deg)

    def ServoJ(self, joint_pos, axisPos, acc=0.0, vel=0.0, cmdT=0.008, filterT=0.0, gain=0.0, id=0, cmdType=0):
        self.controller.servo()
        self.q_deg = list(joint_pos)
        return 0

    def ServoMoveStart(self, cmdType=0):
        return 0

    def ServoMoveEnd(self, cmdType=0):
        return 0

    @property
    def robot_state_pkg(self):  # a fresh packet every time, like the 20004 stream
        c = self.controller
        return SimpleNamespace(
            jt_cur_pos=list(self.q_deg), actual_qd=[0.0] * 6, EmergencyStop=0, main_code=0, sub_code=0,
            safety_stop0_state=0, safety_stop1_state=0, collisionState=0, robot_state=1, robot_mode=0,
            gripper_position=0, gripper_active=1 << 2, mc_queue_len=c.queue,
            gripper_motiondone=0 if c.gripper_moving() else 3,
        )


class WholeChainTests(unittest.TestCase):
    def chain(self, **gripper_options):
        controller = QueueController()
        self.addCleanup(controller.close)
        cfg = Config()
        cfg.fairino.ip = "127.0.0.1"
        cfg.fairino.rpc_port = controller.port
        cfg.fairino.rpc_timeout_s = 2.0
        cfg.fairino.gripper.min_interval_s = 0.05
        cfg.fairino.gripper.pause_cooldown_s = 0.2
        for key, value in gripper_options.items():
            setattr(cfg.fairino.gripper, key, value)
        cfg.validate()
        model = RobotModel(cfg.robot)
        robot = FairinoRobot(cfg, model.lower, model.upper, sdk=object())
        robot._rpc = StreamingRPC(controller, cfg.robot.home_deg)
        robot._connected = True
        robot._bind_sdk_calls()
        robot._prepare_gripper()  # spawns the real gripper process
        self.addCleanup(robot._disconnect)
        self.addCleanup(robot.stop_streaming)
        robot.start_streaming()  # waits for the gripper activation
        return controller, robot

    @staticmethod
    def moves(controller):
        return [a[1] for n, a in controller.calls if n == "MoveGripper"]

    def test_gripper_moves_go_through_and_leave_no_lag(self):
        controller, robot = self.chain()
        q0 = robot.get_state().q.copy()
        robot.set_joint_target(q0 + math.radians(3.0))  # the operator moves the arm meanwhile
        self.assertTrue(wait_for(lambda: self.moves(controller), 10.0))  # the start-up sync
        for wanted in (100, 0, 100):
            robot.set_gripper(wanted / 100.0)
            self.assertTrue(
                wait_for(lambda: self.moves(controller)[-1:] == [wanted], 10.0), (wanted, self.moves(controller))
            )
            self.assertTrue(wait_for(lambda: not robot.paused, 3.0))
        self.assertEqual(controller.faults, [])
        self.assertEqual(controller.queued_behind_gripper, 0)  # nothing was sent while the gripper moved
        self.assertLessEqual(controller.max_queue_after_start, robot.max_backlog_points)  # so no lag built up
        self.assertGreaterEqual(robot.pauses, 3)
        count = controller.servo_count
        time.sleep(0.2)
        self.assertGreater(controller.servo_count, count)  # and the arm streams again afterwards
        self.assertLessEqual(controller.queue, 3)
        self.assertTrue(robot.streaming)
        self.assertEqual(robot._gripper_process.status_text().split(" (")[0], "gripper process running")

    def test_without_the_pause_the_latency_guard_still_removes_the_lag(self):
        controller, robot = self.chain(pause_arm_for_moves=False)
        self.assertTrue(wait_for(lambda: self.moves(controller), 10.0))  # the start-up move: taken at once ...
        self.assertTrue(wait_for(lambda: controller.queued_behind_gripper > 10, 2.0))  # ... set-points pile up
        self.assertTrue(wait_for(lambda: robot.drain_ticks > 0, 2.0))
        time.sleep(QueueController.GRIPPER_S + 0.3)
        self.assertLessEqual(controller.queue, robot.max_backlog_points + 1)  # drained back: lag bounded
        self.assertLessEqual(controller.max_queue_after_start, robot.max_backlog_points + 2)


class FairinoPauseHookTests(unittest.TestCase):
    def robot(self, **packet):
        cfg = Config().validate()
        robot = FairinoRobot(cfg, [-3.0] * 6, [3.0] * 6, sdk=object())
        robot._rpc = SimpleNamespace(robot_state_pkg=SimpleNamespace(**packet))
        return robot

    def test_silent_until_the_gripper_move_is_done_and_the_queue_is_empty(self):
        self.assertFalse(self.robot(gripper_motiondone=3, mc_queue_len=0)._pause_can_end(0.05))  # not yet visible
        self.assertFalse(self.robot(gripper_motiondone=0, mc_queue_len=0)._pause_can_end(0.5))  # still moving
        self.assertFalse(self.robot(gripper_motiondone=3, mc_queue_len=4)._pause_can_end(0.5))  # queue not empty
        self.assertTrue(self.robot(gripper_motiondone=3, mc_queue_len=0)._pause_can_end(0.5))
        self.assertTrue(self.robot(gripper_motiondone=0, mc_queue_len=0)._pause_can_end(3.0))  # gripper time limit

    def test_backlog_and_defaults(self):
        robot = self.robot(mc_queue_len=7)
        self.assertEqual(robot._command_backlog(), 7)
        self.assertEqual(robot.max_backlog_points, Config().fairino.max_queue_points)
        self.assertEqual(robot.pause_timeout_s, Config().fairino.gripper.pause_timeout_s)


if __name__ == "__main__":
    unittest.main()
