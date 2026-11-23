"""Orbbec RGB capture for the headset and episode recorder.

The optional ``pyorbbecsdk`` module is loaded only when a camera is opened. The
SDK v2 distribution is named ``pyorbbecsdk2``; it supports the Gemini 2. This
module deliberately captures color only: depth needs a separate lossless,
calibrated transport rather than the JPEG protocol used by the headset.

API reference: https://orbbec.github.io/pyorbbecsdk/
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import OrbbecConfig

logger = logging.getLogger(__name__)

COLOR_FORMATS = ("MJPG", "RGB", "BGR", "YUYV", "UYVY", "I420", "NV12", "NV21")


class CameraError(RuntimeError):
    """A camera cannot be opened or supplied an invalid frame."""


def load_sdk():
    """Import the native SDK without making it a simulation dependency."""
    try:
        return importlib.import_module("pyorbbecsdk")
    except (ImportError, OSError) as exc:
        raise CameraError(
            "Orbbec SDK is unavailable. In the camera environment run "
            "'python -m pip install pyorbbecsdk2'. The Python import is pyorbbecsdk. "
            "See https://github.com/orbbec/pyorbbecsdk for supported firmware and USB permissions."
        ) from exc


def format_name(pixel_format) -> str:
    return str(getattr(pixel_format, "name", str(pixel_format).rsplit(".", 1)[-1]))


def frame_to_bgr(frame) -> np.ndarray:
    """Decode a color frame into an owned, contiguous uint8 BGR image.

    Reject truncated buffers instead of resizing/repeating data, which could
    turn a damaged SDK frame into an apparently valid recording.
    """
    width, height = int(frame.get_width()), int(frame.get_height())
    name = format_name(frame.get_format())
    if width <= 0 or height <= 0:
        raise CameraError(f"invalid Orbbec frame size {width}x{height}")
    try:
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if name == "MJPG":
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None or image.shape != (height, width, 3):
                raise CameraError("cannot decode Orbbec MJPG frame or frame size does not match its profile")
        elif name in ("RGB", "BGR"):
            image = data.reshape(height, width, 3)
            if name == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif name in ("YUYV", "UYVY"):
            conversion = cv2.COLOR_YUV2BGR_YUY2 if name == "YUYV" else cv2.COLOR_YUV2BGR_UYVY
            image = cv2.cvtColor(data.reshape(height, width, 2), conversion)
        elif name in ("I420", "NV12", "NV21"):
            conversion = {
                "I420": cv2.COLOR_YUV2BGR_I420,
                "NV12": cv2.COLOR_YUV2BGR_NV12,
                "NV21": cv2.COLOR_YUV2BGR_NV21,
            }[name]
            image = cv2.cvtColor(data.reshape(height * 3 // 2, width), conversion)
        else:
            raise CameraError(f"unsupported Orbbec color format: {name}")
    except (TypeError, ValueError, cv2.error) as exc:
        raise CameraError(f"invalid {name} color buffer for {width}x{height}: {exc}") from exc
    # The SDK may recycle its frame buffer after this function returns.
    return np.array(image, dtype=np.uint8, order="C", copy=True)


def list_devices(sdk=None) -> list[dict]:
    """Return connected camera names and serials without starting any streams."""
    sdk = load_sdk() if sdk is None else sdk
    context = sdk.Context()
    devices = context.query_devices()
    return [
        {
            "name": devices.get_device_name_by_index(i),
            "serial": devices.get_device_serial_number_by_index(i),
        }
        for i in range(devices.get_count())
    ]


def select_color_profile(profiles, width: int, height: int, fps: int):
    """Select an exact resolution/rate with a decodable pixel format.

    MJPG is preferred for USB bandwidth. Do not silently use a default profile:
    the operator should know if the requested dimensions/rate are unavailable.
    """
    candidates, available = [], []
    for i in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
        name = format_name(profile.get_format())
        dims = (profile.get_width(), profile.get_height(), profile.get_fps())
        available.append(f"{dims[0]}x{dims[1]}@{dims[2]} {name}")
        if dims == (width, height, fps) and name in COLOR_FORMATS:
            candidates.append((COLOR_FORMATS.index(name), profile))
    if not candidates:
        raise CameraError(
            f"no supported color profile for {width}x{height}@{fps}. "
            f"Available: {', '.join(available) or 'none'}. "
            "Set orbbec.width, orbbec.height and orbbec.fps to a supported combination."
        )
    return min(candidates, key=lambda item: item[0])[1]


class OrbbecCamera:
    """One camera pipeline. Constructing this object does not open hardware."""

    def __init__(self, cfg: OrbbecConfig, sdk=None):
        self.cfg = cfg
        self._sdk = sdk
        self._context = None
        self._device = None
        self._pipeline = None
        self.profile = None
        self.device_name = ""
        self.serial = ""

    def start(self) -> OrbbecCamera:
        if self._pipeline is not None:
            return self
        sdk = load_sdk() if self._sdk is None else self._sdk
        self._sdk = sdk
        try:
            self._context = sdk.Context()
            devices = self._context.query_devices()
            if devices.get_count() == 0:
                raise CameraError("no Orbbec camera found; check the USB connection and SDK USB permissions")
            if self.cfg.serial:
                serials = [devices.get_device_serial_number_by_index(i) for i in range(devices.get_count())]
                if self.cfg.serial not in serials:
                    raise CameraError(f"Orbbec serial {self.cfg.serial!r} not found (connected: {', '.join(serials)})")
                self._device = devices.get_device_by_serial_number(self.cfg.serial)
            else:
                if devices.get_count() != 1:
                    raise CameraError("multiple Orbbec cameras found; select one with --serial or orbbec.serial")
                self._device = devices.get_device_by_index(0)
            info = self._device.get_device_info()
            self.device_name, self.serial = info.get_name(), info.get_serial_number()
            pipeline = sdk.Pipeline(self._device)
            profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
            self.profile = select_color_profile(profiles, self.cfg.width, self.cfg.height, self.cfg.fps)
            stream_cfg = sdk.Config()
            stream_cfg.enable_stream(self.profile)
            self._pipeline = pipeline
            pipeline.start(stream_cfg)
        except Exception as exc:
            self.close()
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"cannot start Orbbec color stream: {exc}") from exc
        return self

    def read(self) -> Optional[np.ndarray]:
        """Read one BGR image, or None when no color frame arrives before timeout."""
        if self._pipeline is None:
            raise CameraError("Orbbec camera is not started")
        try:
            frames = self._pipeline.wait_for_frames(self.cfg.frame_timeout_ms)
            color = None if frames is None else frames.get_color_frame()
            return None if color is None else frame_to_bgr(color)
        except CameraError:
            raise
        except Exception as exc:
            raise CameraError(f"cannot read Orbbec color stream: {exc}") from exc

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                logger.warning("while stopping the Orbbec camera: %s", exc)
        self._device = None
        self._context = None

    def __enter__(self) -> OrbbecCamera:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()
