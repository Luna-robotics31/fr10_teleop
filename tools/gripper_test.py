#!/usr/bin/env python3
"""Open / close the real FR10 gripper from the command line (no arm motion).

    python tools/gripper_test.py open
    python tools/gripper_test.py close
    python tools/gripper_test.py cycle            # open, close, open

Uses the same calls as the teleoperation program (``fairino.gripper`` in config/default.yaml:
index 2, speed 100, force 67, 0 % = open, 100 % = closed). Connecting enables the arm, as in
teleoperation, but no joint command is sent. Keep the gripper area clear and the e-stop within reach.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fr10.config import ConfigError, load_config, parse_override  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("open", "close", "cycle"))
    parser.add_argument("--config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--robot-ip")
    parser.add_argument("--sdk-path")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between moves in 'cycle'")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        overrides = dict(parse_override(v) for v in args.overrides)
        if args.robot_ip:
            overrides["fairino.ip"] = args.robot_ip
        if args.sdk_path:
            overrides["fairino.sdk_path"] = args.sdk_path
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        parser.error(str(exc))

    from fr10.kinematics import RobotModel
    from fr10.robots import RobotError, create_robot

    model = RobotModel(cfg.robot)
    robot = create_robot("real", cfg, model.lower, model.upper)
    try:
        robot.connect()
        steps = {"open": [robot.open_gripper], "close": [robot.close_gripper],
                 "cycle": [robot.open_gripper, robot.close_gripper, robot.open_gripper]}[args.action]
        for index, step in enumerate(steps):
            if index:
                time.sleep(args.pause)
            step()
        return 0
    except RobotError as exc:
        logging.error("%s", exc)
        return 1
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
