"""Camera tests use synthetic buffers and an SDK stand-in; no USB device is opened."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import camera_fr10
from fr10.orbbec_camera import CameraError, OrbbecCamera, frame_to_bgr, list_devices, select_color_profile


class Frame:
    def __init__(self, data, pixel_format="RGB", width=2, height=2):
        self.data, self.pixel_format, self.width, self.height = data, pixel_format, width, height

    def get_data(self):
        return self.data

    def get_format(self):
        return self.pixel_format

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height


class Profile(Frame):
    def __init__(self, pixel_format="RGB", width=640, height=480, fps=30):
        super().__init__(None, pixel_format, width, height)
        self.fps = fps

    def get_fps(self):
        return self.fps

    def as_video_stream_profile(self):
        return self


class ProfileList:
    def __init__(self, *profiles):
        self.profiles = profiles

    def get_count(self):
        return len(self.profiles)

    def get_stream_profile_by_index(self, index):
        return self.profiles[index]


def settings(**overrides):
    values = dict(serial=None, width=640, height=480, fps=30, frame_timeout_ms=1000, jpeg_quality=80)
    return SimpleNamespace(**(values | overrides))


def sdk_standin(serials=("gemini-123",), fail_start=False):
    devices = Mock()
    devices.get_count.return_value = len(serials)
    devices.get_device_serial_number_by_index.side_effect = lambda i: serials[i]
    devices.get_device_name_by_index.return_value = "Orbbec Gemini 2"
    info = Mock()
    info.get_name.return_value = "Orbbec Gemini 2"
    info.get_serial_number.return_value = serials[0] if serials else ""
    device = Mock()
    device.get_device_info.return_value = info
    devices.get_device_by_index.return_value = device
    devices.get_device_by_serial_number.return_value = device
    context = Mock()
    context.query_devices.return_value = devices
    pipeline = Mock()
    pipeline.get_stream_profile_list.return_value = ProfileList(Profile())
    pipeline.wait_for_frames.return_value = None
    if fail_start:
        pipeline.start.side_effect = RuntimeError("USB stream failed")
    sdk = SimpleNamespace(
        Context=Mock(return_value=context), Pipeline=Mock(return_value=pipeline),
        Config=Mock(), OBSensorType=SimpleNamespace(COLOR_SENSOR="color"),
    )
    return sdk, pipeline, devices


class ColorConversionTests(unittest.TestCase):
    def test_rgb_is_converted_and_bgr_is_preserved_with_owned_memory(self):
        data = np.array([[[1, 2, 3], [10, 20, 30]], [[4, 5, 6], [40, 50, 60]]], dtype=np.uint8)
        rgb_result = frame_to_bgr(Frame(data))
        bgr_result = frame_to_bgr(Frame(data, "BGR"))
        np.testing.assert_array_equal(rgb_result, data[:, :, ::-1])
        np.testing.assert_array_equal(bgr_result, data)
        data[:] = 0
        self.assertEqual(bgr_result[0, 0].tolist(), [1, 2, 3])
        self.assertTrue(bgr_result.flags.c_contiguous)

    def test_mjpg_decodes_in_bgr_order(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[:, :] = [20, 50, 220]
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        result = frame_to_bgr(Frame(encoded.tobytes(), "MJPG", 16, 12))
        np.testing.assert_allclose(result, image, atol=3)

    def test_yuv_formats_decode_black_and_white(self):
        for name, packed in (("YUYV", [16, 128, 235, 128]), ("UYVY", [128, 16, 128, 235])):
            with self.subTest(format=name):
                result = frame_to_bgr(Frame(bytes(packed), name, 2, 1))
                np.testing.assert_allclose(result[0, 0], [0, 0, 0], atol=1)
                np.testing.assert_allclose(result[0, 1], [255, 255, 255], atol=1)
        for name in ("I420", "NV12", "NV21"):
            with self.subTest(format=name):
                result = frame_to_bgr(Frame(bytes([16, 16, 16, 16, 128, 128]), name))
                np.testing.assert_array_equal(result, np.zeros((2, 2, 3), dtype=np.uint8))

    def test_bad_buffers_and_unknown_formats_are_rejected(self):
        for frame in (
            Frame(bytes(11)), Frame(b"bad-jpeg", "MJPG"),
            Frame(bytes(12), "H264"), Frame(b"", width=0),
        ):
            with self.subTest(format=frame.pixel_format), self.assertRaises(CameraError):
                frame_to_bgr(frame)


class CameraLifecycleTests(unittest.TestCase):
    def test_profile_selection_preserves_requested_size_and_prefers_mjpg(self):
        mjpg = Profile("MJPG")
        profiles = ProfileList(Profile("RGB"), Profile("MJPG", width=1920, height=1080), mjpg)
        self.assertIs(select_color_profile(profiles, 640, 480, 30), mjpg)
        with self.assertRaisesRegex(CameraError, "Available:.*640x480@30"):
            select_color_profile(profiles, 1280, 720, 60)

    def test_list_devices_does_not_start_a_pipeline(self):
        sdk, pipeline, _devices = sdk_standin()
        self.assertEqual(list_devices(sdk), [{"name": "Orbbec Gemini 2", "serial": "gemini-123"}])
        sdk.Pipeline.assert_not_called()
        pipeline.start.assert_not_called()

    def test_start_read_and_close_use_selected_device_and_timeout(self):
        sdk, pipeline, devices = sdk_standin()
        camera = OrbbecCamera(settings(serial="gemini-123", frame_timeout_ms=123), sdk)
        with camera:
            devices.get_device_by_serial_number.assert_called_once_with("gemini-123")
            self.assertIsNone(camera.read())
            pipeline.wait_for_frames.assert_called_once_with(123)
            frames = Mock()
            frames.get_color_frame.return_value = Frame(bytes([255, 0, 0] * 4))
            pipeline.wait_for_frames.return_value = frames
            np.testing.assert_array_equal(camera.read()[0, 0], [0, 0, 255])
        camera.close()
        pipeline.stop.assert_called_once()

    def test_device_errors_are_actionable_without_starting(self):
        for serials, config, message in (
            ((), settings(), "no Orbbec camera"),
            (("a", "b"), settings(), "multiple Orbbec cameras"),
            (("a",), settings(serial="missing"), "not found"),
        ):
            with self.subTest(serials=serials):
                sdk, pipeline, _devices = sdk_standin(serials)
                with self.assertRaisesRegex(CameraError, message):
                    OrbbecCamera(config, sdk).start()
                pipeline.start.assert_not_called()

    def test_failed_start_cleans_up_and_missing_sdk_explains_install(self):
        sdk, pipeline, _devices = sdk_standin(fail_start=True)
        with self.assertRaisesRegex(CameraError, "USB stream failed"):
            OrbbecCamera(settings(), sdk).start()
        pipeline.stop.assert_called_once()
        with patch("fr10.orbbec_camera.importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(CameraError, "pip install pyorbbecsdk2"):
                OrbbecCamera(settings()).start()


class PublisherLoopTests(unittest.TestCase):
    def make_cfg(self):
        return SimpleNamespace(image=SimpleNamespace(binocular=False, port=55555), orbbec=settings())

    def test_bounded_capture_publishes_frames_and_releases_resources(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch.object(camera_fr10, "OrbbecCamera") as camera_type, patch.object(camera_fr10, "JpegPublisher") as pub_type:
            camera = camera_type.return_value.__enter__.return_value
            camera.read.return_value = image
            camera.device_name, camera.serial = "Gemini 2", "a"
            camera.profile = Profile()
            self.assertEqual(camera_fr10.run_camera(self.make_cfg(), max_frames=3), 3)
            self.assertEqual(pub_type.return_value.publish.call_count, 3)
            camera_type.return_value.__exit__.assert_called_once()
            pub_type.return_value.close.assert_called_once()

    def test_repeated_timeouts_terminate_and_release_resources(self):
        with patch.object(camera_fr10, "OrbbecCamera") as camera_type, patch.object(camera_fr10, "JpegPublisher") as pub_type:
            camera = camera_type.return_value.__enter__.return_value
            camera.read.return_value = None
            camera.device_name, camera.serial = "Gemini 2", "a"
            camera.profile = Profile()
            with self.assertRaisesRegex(CameraError, "no color frames after 5"):
                camera_fr10.run_camera(self.make_cfg(), max_frames=1)
            self.assertEqual(camera.read.call_count, 5)
            camera_type.return_value.__exit__.assert_called_once()
            pub_type.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
