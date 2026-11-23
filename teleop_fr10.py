#!/usr/bin/env python3
"""Launch FR10 headset teleoperation, or a bounded mock/simulation demo.

Run in the teleoperation environment (``conda activate xr``), not Isaac Lab::

    python teleop_fr10.py --robot sim
    python teleop_fr10.py --robot mock --input scripted --auto-start --duration 5 --no-images
    python teleop_fr10.py --robot real --config config/orbbec_gemini2.yaml --record
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import logging
import math
import signal
from typing import Optional, Sequence

from fr10.config import ConfigError, load_config, parse_override

logger = logging.getLogger("fr10.launcher")

KEY_EVENTS = {
    "r": "start", " ": "stop", "o": "open_gripper", "g": "close_gripper", "h": "home", "c": "record", "x": "reset", "q": "quit",
    "+": "faster", "=": "faster", "-": "slower",
}


def positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError("duration must be finite and greater than zero")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=("sim", "real", "mock"), default="sim")
    parser.add_argument("--input", choices=("xr", "scripted"), default="xr",
                        help="headset input, or a synthetic motion demo for sim/mock")
    parser.add_argument("--config", help="YAML configuration (default: config/default.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--record", action="store_true", help="enable episode recording; toggle with c or controller A")
    parser.add_argument("--display-mode", choices=("auto", "immersive", "ego", "pass-through"))
    parser.add_argument("--no-images", action="store_true", help="disable image reception, display and image recording")
    parser.add_argument("--auto-start", action="store_true", help="skip the r/start gate (sim/mock only)")
    parser.add_argument("--duration", type=positive_seconds, help="exit after this many seconds of control-loop execution")
    parser.add_argument("--robot-ip", help="override fairino.ip")
    parser.add_argument("--sdk-path", help="directory containing the FAIRINO Robot.py or Robot.*.so")
    parser.add_argument("--home-on-start", action="store_true",
                        help="after r, first move the arm to the home (ready) pose (this is the default)")
    parser.add_argument("--no-home-on-start", action="store_true",
                        help="after r, hold the arm where it is instead of moving it to the home pose")
    parser.add_argument("--no-gripper", action="store_true",
                        help="do not touch the gripper (run tools/pinch_gripper.py --from-teleop for it)")
    parser.add_argument("--reset-errors", action="store_true", help="clear controller errors on connection (real robot)")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def configure(args, parser):
    overrides = dict(parse_override(value) for value in args.overrides)
    for key, value in (("xr.display_mode", args.display_mode), ("fairino.ip", args.robot_ip),
                       ("fairino.sdk_path", args.sdk_path)):
        if value is not None:
            overrides[key] = value
    if args.no_images:
        overrides["image.enabled"] = False
    if args.auto_start:
        overrides["control.auto_start"] = True
    if args.home_on_start and args.no_home_on_start:
        parser.error("--home-on-start and --no-home-on-start contradict each other")
    if args.home_on_start:
        overrides["control.home_on_start"] = True
    if args.no_home_on_start:
        overrides["control.home_on_start"] = False
    if args.no_gripper:
        overrides["gripper.enabled"] = False
    if args.reset_errors:
        overrides["fairino.reset_errors_on_connect"] = True
    cfg = load_config(args.config, overrides)
    if args.robot == "real" and (args.input == "scripted" or cfg.control.auto_start):
        parser.error("the real robot requires --input xr and manual start with r; scripted/auto-start is for sim/mock")
    if not cfg.image.enabled:
        # Immersive/ego require frames. Pass-through still accepts headset motion.
        cfg.xr.display_mode = "pass-through"
    elif cfg.xr.display_mode == "auto":
        cfg.xr.display_mode = "immersive" if args.robot == "sim" else "pass-through"
    return cfg


def run(args, cfg) -> None:
    # Keep heavy/optional imports after argument validation, including --help.
    from fr10.image_stream import JpegSubscriber
    from fr10.keyboard import KeyListener
    from fr10.kinematics import IKSolver, RobotModel
    from fr10.recorder import EpisodeRecorder
    from fr10.robots import create_robot
    from fr10.teleop import Teleoperator
    from fr10.xr_input import ScriptedInput, TeleVuerInput

    model = RobotModel(cfg.robot)
    ik = IKSolver(model, cfg.ik)
    # Finish input/dependency setup before connecting to a real controller, whose
    # connection routine can enable the arm and activate its gripper.
    with ExitStack() as resources:
        source = ScriptedInput() if args.input == "scripted" else TeleVuerInput(cfg, cfg.xr.display_mode)
        resources.callback(source.close)
        images = None
        if cfg.image.enabled and (source.wants_images or (args.record and cfg.recording.save_images)):
            images = JpegSubscriber(cfg.image.host, cfg.image.port)
            resources.callback(images.close)
            logger.info("camera stream: tcp://%s:%d", cfg.image.host, cfg.image.port)
        hand_publisher = None
        if args.input == "xr" and cfg.xr.input_mode == "hand" and cfg.xr.hand_bus_port > 0:
            try:
                from fr10.hand_bus import HandBusPublisher

                hand_publisher = HandBusPublisher(cfg.xr.hand_bus_port)
                resources.callback(hand_publisher.close)
                logger.info(
                    "hand data is shared on 127.0.0.1:%d (run: python tools/pinch_gripper.py --from-teleop)",
                    cfg.xr.hand_bus_port,
                )
            except RuntimeError as exc:
                logger.warning("hand data is not shared: %s", exc)
        robot = create_robot(args.robot, cfg, model.lower, model.upper)
        recorder = EpisodeRecorder(cfg, robot.name) if args.record else None
        if recorder is not None:
            resources.callback(recorder.close)
        resources.callback(robot.close)
        robot.connect()
        teleop = Teleoperator(
            cfg, robot, model, ik, source, recorder=recorder, images=images, hand_publisher=hand_publisher
        )

        def on_key(key: str) -> None:
            event = KEY_EVENTS.get(key.lower())
            if event is not None:
                teleop.post(event)

        keys = KeyListener(on_key)
        resources.callback(keys.stop)
        keys.start()
        previous = signal.signal(signal.SIGTERM, lambda *_: teleop.post("quit"))
        resources.callback(signal.signal, signal.SIGTERM, previous)
        logger.info("%s robot | %s input | display %s", args.robot, args.input, cfg.xr.display_mode)
        logger.info("keys: r start/recover | SPACE stop arm | o open gripper | g close gripper | h home | c record | x reset scene | +/- speed | q quit")
        if not cfg.control.auto_start:
            logger.info(
                "IDLE: press r in this terminal to start streaming; then %s",
                "make a fist with your left hand to make the arm follow your right hand" if cfg.xr.input_mode == "hand"
                else "release and squeeze the grip button to move",
            )
        teleop.run(duration=args.duration)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = configure(args, parser)
    except ConfigError as exc:
        parser.error(str(exc))
    try:
        run(args, cfg)
    except KeyboardInterrupt:
        logger.info("interrupted; stopped teleoperation")
        return 130
    except ImportError as exc:
        logger.error("missing/incompatible dependency: %s. Use the xr environment; see fr10_teleop/README.md", exc)
        return 1
    except Exception as exc:
        logger.error("teleoperation stopped: %s", exc, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
