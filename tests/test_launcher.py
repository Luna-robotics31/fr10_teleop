"""Launcher validation, resource cleanup, and a bounded hardware-free CLI run."""

from contextlib import contextmanager, ExitStack, redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import teleop_fr10 as launcher


class LauncherConfigurationTests(unittest.TestCase):
    def configure(self, *argv):
        parser = launcher.build_parser()
        return launcher.configure(parser.parse_args(argv), parser)

    def assert_rejected_before_run(self, argv):
        with patch.object(launcher, "run") as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                launcher.main(argv)
        self.assertEqual(stopped.exception.code, 2)
        run.assert_not_called()

    def test_real_robot_rejects_scripted_input_and_auto_start_flags(self):
        for extra in (["--input", "scripted"], ["--auto-start"]):
            with self.subTest(extra=extra):
                self.assert_rejected_before_run(["--robot", "real", *extra])

    def test_real_robot_rejects_auto_start_from_override(self):
        self.assert_rejected_before_run(["--robot", "real", "--set", "control.auto_start=true"])

    def test_real_robot_rejects_auto_start_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "automatic.yaml"
            config.write_text("control:\n  auto_start: true\n", encoding="utf-8")
            self.assert_rejected_before_run(["--robot", "real", "--config", str(config)])

    def test_display_auto_resolution_and_explicit_camera_display(self):
        for robot, expected in (("sim", "immersive"), ("mock", "pass-through"), ("real", "pass-through")):
            with self.subTest(robot=robot):
                self.assertEqual(self.configure("--robot", robot).xr.display_mode, expected)
        self.assertEqual(self.configure("--robot", "real", "--display-mode", "ego").xr.display_mode, "ego")
        self.assertEqual(self.configure("--display-mode", "immersive").xr.display_mode, "immersive")

    def test_no_images_uses_pass_through_even_when_camera_display_was_selected(self):
        cfg = self.configure("--display-mode", "ego", "--no-images")
        self.assertFalse(cfg.image.enabled)
        self.assertEqual(cfg.xr.display_mode, "pass-through")
        cfg = self.configure("--set", "image.enabled=false", "--display-mode", "immersive")
        self.assertEqual(cfg.xr.display_mode, "pass-through")

    def test_no_gripper_leaves_the_gripper_to_the_pinch_tool(self):
        self.assertTrue(self.configure("--robot", "real").gripper.enabled)
        cfg = self.configure("--robot", "real", "--no-gripper", "--set", "xr.input_mode=hand")
        self.assertFalse(cfg.gripper.enabled)
        self.assertEqual(cfg.xr.hand_bus_port, 5560)  # the hand data is shared for tools/pinch_gripper.py

    def test_invalid_duration_is_rejected_before_run(self):
        for value in ("0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(value=value):
                self.assert_rejected_before_run(["--duration", value])

    def test_bad_config_and_overrides_are_rejected_before_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_rejected_before_run(["--config", str(Path(tmp) / "missing.yaml")])
            config = Path(tmp) / "invalid.yaml"
            config.write_text("control:\n  rate_hz: fast\n", encoding="utf-8")
            self.assert_rejected_before_run(["--config", str(config)])
        self.assert_rejected_before_run(["--set", "nonexistent.value=1"])
        self.assert_rejected_before_run(["--set", "control.rate_hz"])


class LauncherResourceTests(unittest.TestCase):
    @contextmanager
    def runtime(self):
        """Replace external boundaries while exercising the launcher's actual wiring."""
        events = []
        source = Mock(wants_images=False)
        images = Mock()
        robot = Mock()
        robot.name = "test_robot"
        recorder = Mock()
        keys = Mock()
        teleop = Mock()
        for name, resource in (("source", source), ("images", images), ("robot", robot), ("recorder", recorder)):
            resource.close.side_effect = lambda name=name: events.append(f"{name}.close")
        keys.stop.side_effect = lambda: events.append("keys.stop")
        boundaries = {
            "source_factory": ("fr10.xr_input.ScriptedInput", source),
            "xr_factory": ("fr10.xr_input.TeleVuerInput", source),
            "subscriber": ("fr10.image_stream.JpegSubscriber", images),
            "robot_factory": ("fr10.robots.create_robot", robot),
            "recorder_factory": ("fr10.recorder.EpisodeRecorder", recorder),
            "keys_factory": ("fr10.keyboard.KeyListener", keys),
            "teleop_factory": ("fr10.teleop.Teleoperator", teleop),
            "model_factory": ("fr10.kinematics.RobotModel", SimpleNamespace(lower=[-1.0] * 6, upper=[1.0] * 6)),
            "ik_factory": ("fr10.kinematics.IKSolver", object()),
        }
        with ExitStack() as patches:
            factories = {name: patches.enter_context(patch(target, return_value=value))
                         for name, (target, value) in boundaries.items()}
            signals = patches.enter_context(patch.object(launcher.signal, "signal", return_value="previous_handler"))
            yield SimpleNamespace(source=source, images=images, robot=robot, recorder=recorder,
                                  keys=keys, teleop=teleop, events=events, signals=signals, **factories)

    def arguments(self, *extra):
        parser = launcher.build_parser()
        args = parser.parse_args(["--robot", "mock", "--input", "scripted", "--duration", "0.1", *extra])
        return args, launcher.configure(args, parser)

    def test_recording_subscribes_images_even_when_scripted_source_has_no_display(self):
        args, cfg = self.arguments("--record")
        with self.runtime() as runtime:
            launcher.run(args, cfg)

        runtime.subscriber.assert_called_once_with(cfg.image.host, cfg.image.port)
        self.assertIs(runtime.teleop_factory.call_args.kwargs["images"], runtime.images)
        self.assertIs(runtime.teleop_factory.call_args.kwargs["recorder"], runtime.recorder)
        runtime.teleop.run.assert_called_once_with(duration=0.1)
        self.assertEqual(runtime.events, ["keys.stop", "robot.close", "recorder.close", "images.close", "source.close"])

    def test_scripted_run_without_image_recording_does_not_subscribe(self):
        for extra in ([], ["--record", "--no-images"], ["--record", "--set", "recording.save_images=false"]):
            with self.subTest(extra=extra):
                args, cfg = self.arguments(*extra)
                with self.runtime() as runtime:
                    launcher.run(args, cfg)
                runtime.subscriber.assert_not_called()

    def test_connection_failure_closes_all_constructed_resources(self):
        args, cfg = self.arguments("--record")
        with self.runtime() as runtime:
            runtime.robot.connect.side_effect = RuntimeError("controller not reachable")
            with self.assertRaisesRegex(RuntimeError, "controller not reachable"):
                launcher.run(args, cfg)

        runtime.keys_factory.assert_not_called()
        runtime.teleop_factory.assert_not_called()
        self.assertEqual(runtime.events, ["robot.close", "recorder.close", "images.close", "source.close"])

    def test_run_failure_closes_resources_and_restores_signal_handler(self):
        args, cfg = self.arguments("--record")
        with self.runtime() as runtime:
            runtime.teleop.run.side_effect = RuntimeError("input failed")
            with self.assertRaisesRegex(RuntimeError, "input failed"):
                launcher.run(args, cfg)

        self.assertEqual(runtime.events, ["keys.stop", "robot.close", "recorder.close", "images.close", "source.close"])
        self.assertEqual(runtime.signals.call_count, 2)
        self.assertEqual(runtime.signals.call_args.args, (launcher.signal.SIGTERM, "previous_handler"))

    def test_input_dependency_failure_happens_before_robot_connection(self):
        parser = launcher.build_parser()
        args = parser.parse_args(["--robot", "real"])
        cfg = launcher.configure(args, parser)
        with self.runtime() as runtime:
            runtime.xr_factory.side_effect = ImportError("headset package missing")
            with self.assertRaisesRegex(ImportError, "headset package missing"):
                launcher.run(args, cfg)

        runtime.robot_factory.assert_not_called()
        runtime.robot.connect.assert_not_called()


class LauncherSmokeTests(unittest.TestCase):
    def test_bounded_mock_scripted_cli_runs_without_headset_or_camera(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-B", str(project / "teleop_fr10.py"), "--robot", "mock", "--input", "scripted",
             "--auto-start", "--duration", "4.0", "--no-images"],
            cwd=project, capture_output=True, text=True, timeout=20.0,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("mock robot | scripted input", output)
        self.assertIn("TRACKING", output)
        self.assertIn("[mock] disconnected", output)


if __name__ == "__main__":
    unittest.main()
