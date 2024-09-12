from unittest.mock import Mock

from gesturemote.gesture_handler import Gesture, GestureHandler


class TestGestureHandler:
    def test_constructor(self):
        gestures = [Gesture(label="test", delay=3, callback=lambda: print("test"))]
        gesture_handler = GestureHandler(gestures=gestures)
        assert gesture_handler.gestures == {"test": gestures[0]}

    def test_handle(self):
        mock_callback = Mock()
        mock_callback.return_value = None

        gestures = [Gesture(label="test", delay=3, callback=mock_callback)]
        gesture_handler = GestureHandler(gestures)
        gesture_handler.handle("test")
        assert gesture_handler.label_queue == ["test"]
        gesture_handler.handle("test")
        assert gesture_handler.label_queue == ["test", "test"]
        gesture_handler.handle("test")
        assert gesture_handler.label_queue == []
        assert mock_callback.call_count == 1
