import click

from gesturemote.gesture_controller import GestureController


@click.group()
def cli():
    """
    GestuReMote command line interface.
    """
    pass


@cli.command()
@click.option("--cursor-sensitivity", type=int, default=7)
@click.option("--scroll-sensitivity", type=float, default=0.1)
@click.option("--activate-gesture-threshold", type=int, default=7)
@click.option("--click-threshold", type=float, default=0.1)
@click.option("--frame-margin", type=float, default=0.1)
@click.option("--monitor-fps", is_flag=True, help="Monitor frames rate.")
@click.option("--verbose", is_flag=True, help="Log verbose output.")
@click.option("--video", is_flag=True, help="Show video stream (experimental).")
def activate(
    cursor_sensitivity: int,
    scroll_sensitivity: float,
    activate_gesture_threshold: int,
    click_threshold: float,
    frame_margin: float,
    monitor_fps: bool,
    verbose: bool,
    video: bool,
):
    """
    Activate GestuReMote.
    """
    gc = GestureController(
        cursor_sensitivity=cursor_sensitivity,
        scroll_sensitivity=scroll_sensitivity,
        activate_gesture_threshold=activate_gesture_threshold,
        click_threshold=click_threshold,
        frame_margin=frame_margin,
        monitor_fps=monitor_fps,
        verbose=verbose,
    )
    gc.activate(video=video)
