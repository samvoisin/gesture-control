# external libraries
import click

# gesturemote library
from gesturemote.gesture_controller import GestureController


@click.group()
def cli():
    """
    GestureMote command line interface.
    """
    pass


@cli.command()
@click.option("--monitor-fps", is_flag=True, help="Monitor frames rate.")
@click.option("--verbose", is_flag=True, help="Log verbose output.")
@click.option("--video-preview", is_flag=True, help="Show video stream (experimental).")
def activate(monitor_fps: bool, verbose: bool, video_preview: bool):
    """
    Activate GestureMote.
    """
    gc = GestureController(monitor_fps=monitor_fps, verbose=verbose)
    gc.activate(video_preview=video_preview)
