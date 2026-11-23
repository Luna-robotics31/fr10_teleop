"""Robot backends: ``mock`` (in-process), ``sim`` (Isaac Lab) and ``real`` (FAIRINO FR10)."""

from __future__ import annotations

from typing import Sequence

from ..config import Config
from .base import Robot, RobotError, RobotState

BACKENDS = ("sim", "real", "mock")


def create_robot(kind: str, cfg: Config, lower: Sequence[float], upper: Sequence[float]) -> Robot:
    """Build the requested backend. Imports are lazy so unused backends cost nothing."""
    if kind == "sim":
        from .sim import SimRobot

        return SimRobot(cfg, lower, upper)
    if kind == "real":
        from .fairino import FairinoRobot

        return FairinoRobot(cfg, lower, upper)
    if kind == "mock":
        from .mock import MockRobot

        return MockRobot(cfg, lower, upper)
    raise ValueError(f"unknown robot backend '{kind}' (choose from {BACKENDS})")


__all__ = ["BACKENDS", "Robot", "RobotError", "RobotState", "create_robot"]
