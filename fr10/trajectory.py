"""Joint set-point shaping.

The inverse kinematics produces a new joint target at the control rate, with whatever
noise and discontinuities the operator's hand has. A robot controller wants a smooth
stream at its servo period. :class:`JointSetpointFilter` sits in between: every servo
tick it moves the commanded position towards the latest target while respecting
per-joint velocity, acceleration and position limits, and it brakes early enough not
to overshoot a target that has stopped moving.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class JointSetpointFilter:
    """Velocity/acceleration limited tracking of a moving joint target.

    Per joint, with tracking error ``e = target - position``::

        v_des = sign(e) * min(v_max, sqrt(a_max * |e|), gain * |e|)
        v    += clip(v_des - v, -a_max * dt, +a_max * dt)
        pos  += v * dt

    ``sqrt(a_max |e|)`` is the speed from which the joint can still stop at the target
    using half of the acceleration limit, which leaves margin for discretisation; the
    ``gain * |e|`` term takes over close to the target and removes chatter.
    """

    def __init__(
        self,
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        gain: float = 15.0,
    ):
        self.max_velocity = np.asarray(max_velocity, dtype=float).copy()
        self.max_acceleration = np.asarray(max_acceleration, dtype=float).copy()
        self.lower = np.asarray(lower, dtype=float).copy()
        self.upper = np.asarray(upper, dtype=float).copy()
        self.gain = float(gain)
        n = self.max_velocity.size
        if not (self.max_acceleration.size == self.lower.size == self.upper.size == n):
            raise ValueError("limit vectors must have the same length")
        if np.any(self.max_velocity <= 0.0) or np.any(self.max_acceleration <= 0.0) or self.gain <= 0.0:
            raise ValueError("velocity, acceleration limits and gain must be positive")
        if np.any(self.lower >= self.upper):
            raise ValueError("lower limits must be below upper limits")
        self._position: Optional[np.ndarray] = None
        self._velocity = np.zeros(n)

    @property
    def initialized(self) -> bool:
        return self._position is not None

    @property
    def position(self) -> np.ndarray:
        if self._position is None:
            raise RuntimeError("JointSetpointFilter.reset() has not been called")
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def reset(self, position: Sequence[float]) -> None:
        """Start from ``position`` at rest (call with the measured joint position)."""
        position = np.asarray(position, dtype=float)
        if position.shape != self.lower.shape or not np.all(np.isfinite(position)):
            raise ValueError("reset position must be a finite vector of the right size")
        self._position = position.copy()
        self._velocity = np.zeros_like(position)

    def stop(self) -> None:
        """Forget any velocity: the next step starts from rest at the current position."""
        self._velocity = np.zeros_like(self._velocity)

    def stopping_point(self, speed_scale: float = 1.0) -> np.ndarray:
        """Where every joint comes to rest if :meth:`step` is given this point as its target.

        ``step`` approaches a target with ``|v| <= sqrt(a_max * distance)``, so a joint moving at
        ``v`` can stop smoothly ``v * |v| / a_max`` further on (within the joint limits).
        """
        if self._position is None:
            raise RuntimeError("JointSetpointFilter.reset() has not been called")
        a_max = self.max_acceleration * float(np.clip(speed_scale, 1e-3, 1.0))
        return np.clip(self._position + self._velocity * np.abs(self._velocity) / a_max, self.lower, self.upper)

    def step(
        self, target: Sequence[float], dt: float, speed_scale: float = 1.0, velocity_scale: float = 1.0
    ) -> np.ndarray:
        """Advance one servo tick towards ``target`` and return the new set-point.

        ``speed_scale`` scales velocity and acceleration limits; ``velocity_scale`` (0..1) scales only the
        velocity limit, so the filter can be slowed down without losing its ability to brake.
        """
        if self._position is None:
            raise RuntimeError("JointSetpointFilter.reset() has not been called")
        if not (dt > 0.0):
            raise ValueError("dt must be positive")
        target = np.asarray(target, dtype=float)
        if target.shape != self._position.shape or not np.all(np.isfinite(target)):
            raise ValueError("target must be a finite vector of the right size")
        scale = float(np.clip(speed_scale, 1e-3, 1.0))
        v_max = self.max_velocity * scale * float(np.clip(velocity_scale, 0.0, 1.0))
        a_max = self.max_acceleration * scale

        target = np.clip(target, self.lower, self.upper)
        error = target - self._position
        distance = np.abs(error)
        v_des = np.sign(error) * np.minimum(np.minimum(v_max, np.sqrt(a_max * distance)), self.gain * distance)
        self._velocity = self._velocity + np.clip(v_des - self._velocity, -a_max * dt, a_max * dt)
        position = self._position + self._velocity * dt

        # Never leave the joint range; stop dead on a limit.
        outside = (position < self.lower) | (position > self.upper)
        if np.any(outside):
            position = np.clip(position, self.lower, self.upper)
            self._velocity[outside] = 0.0
        self._position = position
        return position.copy()


def min_jerk(fraction: float) -> float:
    """Smooth 0 -> 1 profile with zero velocity and acceleration at both ends."""
    s = float(np.clip(fraction, 0.0, 1.0))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


class JointMove:
    """Straight-line joint-space move with a smooth time profile (used for 'go home')."""

    # peak speed of the min-jerk profile is 1.875x its mean speed
    _PEAK_FACTOR = 1.875

    def __init__(self, start: Sequence[float], goal: Sequence[float], max_speed: float, min_duration: float = 1.0):
        self.start = np.asarray(start, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        travel = float(np.max(np.abs(self.goal - self.start)))
        self.duration = max(min_duration, self._PEAK_FACTOR * travel / max_speed)
        self.elapsed = 0.0

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    def step(self, dt: float) -> np.ndarray:
        """Advance time by ``dt`` and return the joint target."""
        self.elapsed = min(self.duration, self.elapsed + max(dt, 0.0))
        return self.start + min_jerk(self.elapsed / self.duration) * (self.goal - self.start)
