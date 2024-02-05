from unittest.mock import MagicMock, patch

import numpy as np

from gesturemote.camera.open_cv import OpenCVCameraInterface


class TestOpenCVCameraInterface:
    @patch("cv2.VideoCapture")
    def test_destructor(self, mock_video_capture):
        mock_camera = MagicMock()
        mock_video_capture.return_value = mock_camera

        camera_interface = OpenCVCameraInterface()
        del camera_interface

        mock_camera.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_get_frame(self, mock_video_capture):
        mock_camera = MagicMock()
        mock_video_capture.return_value = mock_camera

        # Simulate camera read return value
        fake_frame = np.zeros((480, 640, 3))
        mock_camera.read.return_value = (True, fake_frame)

        camera_interface = OpenCVCameraInterface()
        frame = camera_interface.get_frame()

        assert np.array_equal(frame, fake_frame)

    @patch("cv2.VideoCapture")
    def test_stream_frames(self, mock_video_capture):
        mock_camera = MagicMock()
        mock_video_capture.return_value = mock_camera

        # Simulate camera read return value
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_camera.read.return_value = (True, fake_frame)

        camera_interface = OpenCVCameraInterface()
        frame_generator = camera_interface.stream_frames()

        # get the first frame from the generator
        first_frame = next(frame_generator)
        assert np.array_equal(first_frame, fake_frame)

        # confirm second frame to ensure the generator continues correctly
        second_frame = next(frame_generator)
        assert np.array_equal(second_frame, fake_frame)
