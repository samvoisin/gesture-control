# gestrol library
from gesturemote.gesture_controller import GestureController
from gesturemote.macos_routines import MACOS_ROUTINES

if __name__ == "__main__":
    gc = GestureController(routines=MACOS_ROUTINES, monitor_fps=True, verbose=True)
    gc.activate(video_preview=True)  # costs about 0.5 FPS
