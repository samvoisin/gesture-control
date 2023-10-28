# external libraries
import click

# gesturemote library
from gesturemote.gesture_controller import GestureController


@click.group()
def cli():
    """
    GestuReMote command line interface.
    """
    pass


@cli.command()
@click.option("--cursor-smoothing-param", type=int, default=5)
@click.option("--activate-gesture-threshold", type=int, default=7)
@click.option("--monitor-fps", is_flag=True, help="Monitor frames rate.")
@click.option("--verbose", is_flag=True, help="Log verbose output.")
@click.option("--video-preview", is_flag=True, help="Show video stream (experimental).")
def activate(
    cursor_smoothing_param: int, activate_gesture_threshold: int, monitor_fps: bool, verbose: bool, video_preview: bool
):
    """
    Activate GestuReMote.
    """
    gc = GestureController(
        cursor_smoothing_param=cursor_smoothing_param,
        activate_gesture_threshold=activate_gesture_threshold,
        monitor_fps=monitor_fps,
        verbose=verbose,
    )
    gc.activate(video_preview=video_preview)