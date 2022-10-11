# external libraries
import pytest

# gestrol library
from gestrol.control_mode_switch import ControlModeSwitch


class TestControlModeSwitch:
    """
    Set of tests for `ControlModeSwitch`. A class to maintain state.
    """

    cms = ControlModeSwitch(control_mode_signal=0)

    @pytest.mark.parametrize(
        ["signal", "status"],
        [(1, False), (0, True), (2, True), (0, False), (3, False)],
    )
    def test_assess_gesture_signal(self, signal: int, status: bool):
        """
        Test to ensure `assess_gesture_signal` works in all cases.
        """
        self.cms.assess_gesture_signal(signal)
        assert self.cms.control_mode == status
