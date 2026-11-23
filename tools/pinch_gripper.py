#!/usr/bin/env python3
"""Pinch with your hand to close the real FR10 gripper. The arm is never commanded.

    python tools/pinch_gripper.py                 # right hand: pinch = closed, release = open
    python tools/pinch_gripper.py --hand left
    python tools/pinch_gripper.py --toggle        # every new pinch flips open / closed
    python tools/pinch_gripper.py --dry-run       # watch the pinch detection, no robot connection

Alone, it owns the headset connection: put on the headset with hand tracking on (controllers down),
open the page this program prints, press Enter VR, and touch thumb and index fingertip together.

Next to the teleoperation, run the teleoperation with --no-gripper in one terminal (it owns the
headset and shares your hand data) and this tool in another:

    python teleop_fr10.py --robot real --no-gripper --set xr.input_mode=hand --set control.speed_scale=0.2
    python tools/pinch_gripper.py --from-teleop

The gripper moves with the calls of gripper.py: ActGripper(2, 1), then
MoveGripper(2, 0 or 100, 100, 67, 2000, 0, 0, 0, 0, 0). Connecting does not enable the arm and
sends no arm command. Keep the gripper area clear.
"""

import argparse
import logging
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fr10.config import ConfigError, load_config, parse_override  # noqa: E402


def local_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))  # no packet is sent; this only selects the outgoing interface
            return probe.getsockname()[0]
    except OSError:
        return "<this-PC-address>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand", choices=("right", "left"), default="right", help="the hand that pinches")
    parser.add_argument("--toggle", action="store_true", help="each new pinch flips the gripper")
    parser.add_argument("--dry-run", action="store_true", help="do not connect to the robot")
    parser.add_argument("--open-first", action="store_true", help="open the gripper once at start")
    parser.add_argument("--gripper-py-style", action="store_true",
                        help="like gripper.py: ActGripper before every move and wait for each move (slow: "
                             "an activation takes seconds)")
    parser.add_argument("--min-interval", type=float, default=0.4,
                        help="proportional: shortest time between gripper commands while the fingers move [s]")
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="proportional: smallest change in percent sent while the fingers move")
    parser.add_argument("--proportional", action="store_true",
                        help="the gripper follows how far you pinch (half pinch = half closed) instead of open/close")
    parser.add_argument("--from-teleop", action="store_true",
                        help="read the hand data from the running teleoperation program instead of the headset")
    parser.add_argument("--teleop-host", default="127.0.0.1", help="PC running the teleoperation (--from-teleop)")
    parser.add_argument("--connection", choices=("auto", "sdk", "plain"), default="auto",
                        help="how to talk to the controller: the vendor SDK (as gripper.py), or plain XML-RPC "
                             "(no second SDK client; the default with --from-teleop)")
    parser.add_argument("--on-mm", type=float, default=30.0, help="pinch starts below this thumb-index distance")
    parser.add_argument("--off-mm", type=float, default=50.0, help="pinch ends above this distance")
    parser.add_argument("--config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--robot-ip")
    parser.add_argument("--sdk-path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        overrides = dict(parse_override(v) for v in args.overrides)
        overrides.update({"xr.input_mode": "hand", "xr.hand": args.hand, "xr.display_mode": "pass-through"})
        if args.robot_ip:
            overrides["fairino.ip"] = args.robot_ip
        if args.sdk_path:
            overrides["fairino.sdk_path"] = args.sdk_path
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        parser.error(str(exc))
    if not 0.0 < args.on_mm < args.off_mm:
        parser.error("need 0 < --on-mm < --off-mm")

    from fr10.pinch_gripper import (
        PinchDetector, PinchGripperController, ProportionalPinchController, ScriptGripper, pinch_distance,
    )
    from fr10.robots.base import RobotError
    from fr10.hand_bus import HandBusSubscriber
    from fr10.xr_input import TeleVuerInput

    class DryRunGripper:
        def move_percent(self, percent, activate=True, block=0, log=True):
            print(f"   [dry run] would move the gripper to {int(round(percent))} %")

        def open_gripper(self):
            print("   [dry run] would call ActGripper(2, 1) and MoveGripper(2, 0, ...)  -> OPEN")

        def close_gripper(self):
            print("   [dry run] would call ActGripper(2, 1) and MoveGripper(2, 100, ...) -> CLOSE")

    gripper = None
    source = None
    try:
        if args.dry_run:
            print("*** DRY RUN: the robot is NOT connected and the gripper will NOT move. "
                  "Start without --dry-run to move the real gripper. ***")
            gripper = DryRunGripper()
        else:
            plain = args.connection == "plain" or (args.connection == "auto" and args.from_teleop)
            print(f"connecting to the controller at {cfg.fairino.ip} ({'plain XML-RPC' if plain else 'vendor SDK'}) ...")
            gripper = ScriptGripper.connect_plain(cfg) if plain else ScriptGripper.connect(cfg)
            if not args.gripper_py_style:
                print("activating the gripper once (this can take several seconds) ...")
                try:
                    gripper.activate()
                except RobotError as exc:
                    print(f"   activation did not complete ({exc}); continuing, the gripper may already be active")
                gripper.activate_each_move = False  # fast moves from here on: no activation, no waiting
                gripper.block = 1
        if args.from_teleop:
            source = HandBusSubscriber(args.teleop_host, cfg.xr.hand_bus_port)
            print(f"\nreading your {args.hand} hand from the teleoperation program at "
                  f"{args.teleop_host}:{cfg.xr.hand_bus_port} (it must run with --set xr.input_mode=hand).")
        else:
            source = TeleVuerInput(cfg, "pass-through")
            address = local_address()
            print(f"\nIn the headset browser open:  https://{address}:8012/?ws=wss://{address}:8012")
            print("accept the certificate warning, press Enter VR, then pinch thumb and index of your "
                  f"{args.hand} hand.")
        print("Ctrl+C to quit.")
        print("mode: " + ("DRY RUN (no robot)" if args.dry_run else f"REAL gripper via {cfg.fairino.ip}") + "\n")
        if args.proportional:
            controller = ProportionalPinchController(
                gripper, open_m=cfg.xr.gripper_pinch_open_m, closed_m=cfg.xr.gripper_pinch_closed_m,
                resolution=args.resolution, min_interval_s=args.min_interval,
            )
        else:
            controller = PinchGripperController(
                gripper,
                PinchDetector(on_m=args.on_mm / 1000.0, off_m=args.off_mm / 1000.0),
                toggle=args.toggle,
                open_first=args.open_first,
            )
        next_print = 0.0
        while True:
            now = time.monotonic()
            positions, age = source.hand_snapshot(args.hand)
            fresh = positions is not None and age is not None and age <= cfg.xr.stale_timeout_s
            distance = pinch_distance(positions) if fresh else None
            controller.update(distance, now)
            if now >= next_print:
                next_print = now + 0.5
                if args.from_teleop and not source.connected:
                    note = "no hand data: is the teleoperation program running with xr.input_mode=hand?"
                elif distance is None:
                    note = "hand not tracked" if positions is None else "tracking data is stale"
                else:
                    if args.proportional:
                        note = f"thumb-index {1e3 * distance:5.1f} mm"
                    else:
                        note = f"thumb-index {1e3 * distance:5.1f} mm  ({'PINCH' if controller.detector.pinched else 'open '})"
                shown = (
                    "unknown" if controller.position is None else f"{controller.position} %"
                ) if args.proportional else (controller.state or "unknown")
                print(f"{args.hand} hand: {note}   gripper: {shown}")
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("\nstopped; the gripper stays as it is")
        return 0
    except RobotError as exc:
        logging.error("%s", exc)
        return 1
    finally:
        if source is not None:
            source.close()
        if isinstance(gripper, ScriptGripper):
            gripper.close_connection()


if __name__ == "__main__":
    raise SystemExit(main())
