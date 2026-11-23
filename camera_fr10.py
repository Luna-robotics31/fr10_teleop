#!/usr/bin/env python3
"""Publish Orbbec Gemini 2 RGB images to the FR10 headset/recording stream."""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

from fr10.config import Config, ConfigError, load_config, parse_override
from fr10.image_stream import JpegPublisher
from fr10.orbbec_camera import CameraError, OrbbecCamera, format_name, list_devices

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration (default: config/default.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--serial", help="select the Orbbec camera by serial number")
    parser.add_argument("--list-devices", action="store_true", help="list cameras without starting a stream")
    parser.add_argument("--preview", action="store_true", help="show local RGB preview; q or Escape stops")
    parser.add_argument("--bind-host", default="*", help="address on which to publish JPEG frames (default: *)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0: until Ctrl+C)")
    return parser


def run_camera(cfg: Config, *, preview: bool = False, max_frames: int = 0, bind_host: str = "*") -> int:
    """Capture and publish color only; never connect to or command the robot."""
    if cfg.image.binocular:
        raise CameraError("a single Gemini 2 color stream is monocular; set image.binocular=false")
    if max_frames < 0:
        raise CameraError("max_frames must be >= 0")
    import cv2

    publisher = JpegPublisher(cfg.image.port, cfg.orbbec.jpeg_quality, bind_host)
    count, missed = 0, 0
    try:
        with OrbbecCamera(cfg.orbbec) as camera:
            logger.info(
                "%s (serial %s): %dx%d@%d %s -> tcp://%s:%d",
                camera.device_name, camera.serial, cfg.orbbec.width, cfg.orbbec.height,
                cfg.orbbec.fps, format_name(camera.profile.get_format()), bind_host, cfg.image.port,
            )
            while not max_frames or count < max_frames:
                image = camera.read()
                if image is None:
                    missed += 1
                    if missed >= 5:
                        raise CameraError("no color frames after 5 waits; check the USB cable and close other camera applications")
                    logger.warning("no color frame within %d ms", cfg.orbbec.frame_timeout_ms)
                    continue
                missed = 0
                publisher.publish(image)
                count += 1
                if preview:
                    cv2.imshow("FR10 Orbbec RGB", image)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        publisher.close()
        if preview:
            cv2.destroyAllWindows()
    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        overrides = dict(parse_override(value) for value in args.overrides)
        if args.serial is not None:
            overrides["orbbec.serial"] = args.serial
        cfg = load_config(args.config, overrides)
        if args.list_devices:
            devices = list_devices()
            for device in devices:
                print(f"{device['name']}  serial={device['serial']}")
            if not devices:
                print("No Orbbec cameras found.")
            return 0
        run_camera(cfg, preview=args.preview, max_frames=args.max_frames, bind_host=args.bind_host)
        return 0
    except KeyboardInterrupt:
        return 0
    except (CameraError, ConfigError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
