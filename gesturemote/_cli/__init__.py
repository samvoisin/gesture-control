from typing import Optional

import click

from gesturemote.camera import OpenCVCameraInterface
from gesturemote.gesture_controller import GestureController, display_video


@click.group()
def cli():
    """
    GestuReMote command line interface.
    """
    pass


@cli.command()
@click.option("--cursor-sensitivity", type=int, default=8)
@click.option("--scroll-sensitivity", type=float, default=0.075)
@click.option("--inverse-scroll", is_flag=True, help="Invert the scroll direction.")
@click.option("--activate-gesture-threshold", type=int, default=7)
@click.option("--click-threshold", type=float, default=0.1)
@click.option("--frame-margin", type=float, default=0.1)
@click.option("--monitor-fps", is_flag=True, help="Monitor frames rate.")
@click.option("--verbose", is_flag=True, help="Log verbose output.")
@click.option("--logfile", type=str, default=None, help="Log file path.")
@click.option("--video", is_flag=True, help="Show video stream (experimental).")
@click.option("--video-image-size", type=int, default=720, help="Size of the video image.")
@click.option("--camera-index", type=int, default=0)
def activate(
    cursor_sensitivity: int,
    scroll_sensitivity: float,
    inverse_scroll: bool,
    activate_gesture_threshold: int,
    click_threshold: float,
    frame_margin: float,
    monitor_fps: bool,
    verbose: bool,
    logfile: Optional[str],
    video: bool,
    video_image_size: int,
    camera_index: int,
):
    """
    Activate GestuReMote.
    """
    camera = OpenCVCameraInterface(index=camera_index)

    gc = GestureController(
        cursor_sensitivity=cursor_sensitivity,
        scroll_sensitivity=scroll_sensitivity,
        inverse_scroll=inverse_scroll,
        activate_gesture_threshold=activate_gesture_threshold,
        click_threshold=click_threshold,
        frame_margin=frame_margin,
        camera=camera,
        monitor_fps=monitor_fps,
        verbose=verbose,
        logfile=logfile,
    )
    for frame, control_model, finger_landmarks in gc.activate():
        if video:
            display_video(frame, control_model, finger_landmarks, preview_img_size=video_image_size)
