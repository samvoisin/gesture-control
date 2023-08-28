# standard libraries
import subprocess
from typing import Callable, Dict


def _run_applescript(script):
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.returncode}\n{e.stderr}"


def _volume_up():
    _run_applescript("set volume output volume (output volume of (get volume settings) + 10)")


def _volume_down():
    _run_applescript("set volume output volume (output volume of (get volume settings) - 10)")


def _mute():
    _run_applescript("set volume output muted true")


def _open_messages_app():
    _run_applescript('tell application "Messages" to activate')


def _open_finder():
    applescript_code = """
    tell application "Finder"
        activate
        set homeFolder to (~)
        set newWindow to make new Finder window to homeFolder
        select newWindow
    end tell
    """
    _run_applescript(applescript_code)


MACOS_ROUTINES: Dict[str, Callable[[], None]] = {
    "02": _volume_down,  # dislike
    "05": _volume_up,  # like
    "08": _open_messages_app,  # one
    "10": _open_finder,  # peace
    "0606": _mute,  # mute, mute
}
