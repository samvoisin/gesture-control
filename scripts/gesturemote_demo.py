# gesturemote library
from gesturemote.gesture_controller import GestureController

if __name__ == "__main__":
    gc = GestureController(monitor_fps=True)
    gc.activate(video_preview=True)
