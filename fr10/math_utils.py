"""Small SO(3)/SE(3) helpers (NumPy only).

Conventions
-----------
* Poses are 4x4 homogeneous matrices ``T_a_b`` that map coordinates of frame ``b``
  into frame ``a``.
* Quaternions are ``[w, x, y, z]``.
* Roll/pitch/yaw follow the URDF convention: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw to a rotation matrix."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def xyzrpy_to_pose(xyzrpy) -> np.ndarray:
    """``[x, y, z, roll, pitch, yaw]`` (metres, radians) to a 4x4 pose."""
    x, y, z, roll, pitch, yaw = (float(v) for v in xyzrpy)
    return make_pose(rpy_to_matrix(roll, pitch, yaw), np.array([x, y, z]))


def pose_inverse(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4)
    rot_t = pose[:3, :3].T
    inv[:3, :3] = rot_t
    inv[:3, 3] = -rot_t @ pose[:3, 3]
    return inv


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Project a nearly-orthonormal 3x3 matrix onto SO(3)."""
    u, _, vt = np.linalg.svd(rotation)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def is_valid_pose(pose, det_tol: float = 1e-2) -> bool:
    """True if ``pose`` is a finite 4x4 matrix whose rotation block is a proper rotation."""
    pose = np.asarray(pose)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return False
    rot = pose[:3, :3]
    if abs(np.linalg.det(rot) - 1.0) > det_tol:
        return False
    return bool(np.allclose(rot @ rot.T, np.eye(3), atol=det_tol))


def log_so3(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to rotation vector (axis * angle), angle in [0, pi]."""
    cos_angle = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-7:
        return 0.5 * vee
    if np.pi - angle < 1e-5:
        # Near pi the antisymmetric part vanishes; recover the axis from the symmetric part.
        sym = (rotation + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(sym), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] < _EPS:
            return np.zeros(3)
        axis = sym[:, k] / axis[k]
        axis /= max(np.linalg.norm(axis), _EPS)
        if np.dot(axis, vee) < 0.0:
            axis = -axis
        return axis * angle
    return vee * (angle / (2.0 * np.sin(angle)))


def exp_so3(rotvec) -> np.ndarray:
    """Rotation vector to rotation matrix (Rodrigues)."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        k = _skew(rotvec)
        return np.eye(3) + k + 0.5 * (k @ k)
    k = _skew(rotvec / angle)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_angle(rotation: np.ndarray) -> float:
    """Magnitude of the rotation in radians."""
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))


def slerp_rotation(rot_a: np.ndarray, rot_b: np.ndarray, fraction: float) -> np.ndarray:
    """Geodesic interpolation from ``rot_a`` (fraction 0) to ``rot_b`` (fraction 1)."""
    return rot_a @ exp_so3(fraction * log_so3(rot_a.T @ rot_b))


def yaw_of(rotation: np.ndarray, fallback: float | None = None) -> float | None:
    """Heading of the rotated x-axis about +z.

    Returns ``fallback`` if the x-axis is (nearly) vertical and the heading is undefined.
    """
    x_axis = rotation[:3, 0]
    if np.hypot(x_axis[0], x_axis[1]) < 0.2:
        return fallback
    return float(np.arctan2(x_axis[1], x_axis[0]))


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to ``[w, x, y, z]`` (w >= 0)."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    quat /= np.linalg.norm(quat)
    return -quat if quat[0] < 0.0 else quat


def quat_to_matrix(quat) -> np.ndarray:
    """``[w, x, y, z]`` to a rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def pose_to_xyzquat(pose: np.ndarray) -> list[float]:
    """4x4 pose to ``[x, y, z, qw, qx, qy, qz]`` (plain floats, JSON friendly)."""
    return [float(v) for v in pose[:3, 3]] + [float(v) for v in matrix_to_quat(pose[:3, :3])]


def pose_error(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[float, float]:
    """Return (translation distance [m], rotation angle [rad]) between two poses."""
    return (
        float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])),
        rotation_angle(pose_a[:3, :3].T @ pose_b[:3, :3]),
    )
