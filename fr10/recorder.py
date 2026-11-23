"""Episode recording for demonstrations / imitation learning.

Layout::

    <directory>/<task>/episode_0001/
        meta.json        robot, joint names, rates, configuration snapshot
        steps.jsonl      one JSON object per recorded step (state, action, poses, image file)
        images/000000.jpg ...

Writing happens on a background thread so that disk I/O never delays the control loop.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

_STOP = object()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class EpisodeRecorder:
    def __init__(self, cfg: Config, robot_name: str):
        self._cfg = cfg
        self._robot_name = robot_name
        self._root = cfg.recording.directory_path / cfg.recording.task
        self._period = 1.0 / cfg.recording.rate_hz
        self._queue: "queue.Queue" = queue.Queue(maxsize=2000)
        self._thread: Optional[threading.Thread] = None
        self._episode_dir: Optional[Path] = None
        self._steps = 0
        self._dropped = 0
        self._t0 = 0.0
        self._next_sample = 0.0
        self.episodes_saved = 0

    @property
    def active(self) -> bool:
        return self._episode_dir is not None

    @property
    def episode_dir(self) -> Optional[Path]:
        return self._episode_dir

    def _next_episode_dir(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        numbers = [
            int(p.name.split("_")[-1])
            for p in self._root.glob("episode_*")
            if p.is_dir() and p.name.split("_")[-1].isdigit()
        ]
        return self._root / f"episode_{max(numbers, default=0) + 1:04d}"

    def start(self) -> Path:
        """Begin a new episode and return its directory."""
        if self.active:
            return self._episode_dir
        episode_dir = self._next_episode_dir()
        (episode_dir / "images").mkdir(parents=True)
        meta = {
            "format": "fr10_teleop/1",
            "robot": self._robot_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rate_hz": self._cfg.recording.rate_hz,
            "joint_names": list(self._cfg.robot.arm_joints),
            "units": {"joints": "rad", "poses": "[x, y, z, qw, qx, qy, qz], m, robot base frame", "gripper": "0 open .. 1 closed"},
            "config": self._cfg.to_dict(),
        }
        (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._episode_dir = episode_dir
        self._steps = 0
        self._dropped = 0
        self._t0 = time.monotonic()
        self._next_sample = self._t0
        self._thread = threading.Thread(target=self._writer, args=(episode_dir,), name="recorder", daemon=True)
        self._thread.start()
        logger.info("recording episode -> %s", episode_dir)
        return episode_dir

    def due(self, now: Optional[float] = None) -> bool:
        """True when the next sample should be taken (keeps the configured recording rate)."""
        if not self.active:
            return False
        now = time.monotonic() if now is None else now
        if now + 1e-6 < self._next_sample:
            return False
        self._next_sample = max(self._next_sample + self._period, now - self._period)
        return True

    def add(self, sample: dict, image: Optional[np.ndarray] = None) -> None:
        """Queue one step. ``image`` is a BGR frame or ``None``."""
        if not self.active:
            return
        record = _jsonable(sample)
        record["step"] = self._steps
        record["t"] = time.monotonic() - self._t0
        if image is not None and self._cfg.recording.save_images:
            record["image"] = f"images/{self._steps:06d}.jpg"
        else:
            image = None
        try:
            self._queue.put_nowait((record, image))
            self._steps += 1
        except queue.Full:
            self._dropped += 1
            if self._dropped in (1, 100) or self._dropped % 1000 == 0:
                logger.warning("recorder cannot keep up with the disk: %d steps dropped", self._dropped)

    def stop(self) -> Optional[Path]:
        """Finish the episode (blocks until everything is on disk)."""
        if not self.active:
            return None
        episode_dir, self._episode_dir = self._episode_dir, None
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            self._thread = None
        self.episodes_saved += 1
        logger.info("episode saved: %d steps in %s", self._steps, episode_dir)
        return episode_dir

    def close(self) -> None:
        self.stop()

    def _writer(self, episode_dir: Path) -> None:
        quality = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._cfg.recording.jpeg_quality)]
        with open(episode_dir / "steps.jsonl", "a", encoding="utf-8") as stream:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                record, image = item
                try:
                    if image is not None and not cv2.imwrite(str(episode_dir / record["image"]), image, quality):
                        record.pop("image", None)
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                except (OSError, ValueError, TypeError) as exc:
                    logger.error("recorder: could not write step %s: %s", record.get("step"), exc)
