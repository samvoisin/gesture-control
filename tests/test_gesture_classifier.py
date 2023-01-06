# standard libraries
from unittest.mock import Mock

# external libraries
import torch

# gestrol library
from gestrol.gesture_classifier import PytorchGestureClassifier


def test_pytorch_gesture_classifier():
    mock_torch_model = Mock(torch.nn.Module)
    pgc = PytorchGestureClassifier(model=mock_torch_model, device=torch.device("cpu"))
    pgc.model.eval.assert_called()
    gest_num = 0
    pgc.model.return_value = torch.Tensor([1.0, 0.0, -1.0])

    frame = torch.empty(10, 10, 3)
    ig = pgc.infer_gesture(frame)

    pgc.model.assert_called_with(frame)
    assert isinstance(ig, int)
    assert gest_num == ig
