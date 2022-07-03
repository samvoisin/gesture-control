# external libraries
import numpy as np
from PIL import Image
from torch import Tensor

# gestrol library
from gestrol.frame_stream.modifiers.base import FrameModifier


class NumpyToTensorModifier(FrameModifier):
    def modify_frame(self, frame: np.ndarray) -> Tensor:
        return Tensor(frame)


class TensorToNumpyModifier(FrameModifier):
    def modify_frame(self, frame: Tensor) -> np.ndarray:
        return frame.detach().numpy()


class NumpyToImageModifier(FrameModifier):
    def modify_frame(self, frame: np.ndarray) -> Image.Image:
        res = Image.fromarray(frame.astype("uint8"), "RGB")
        res.save("/home/samvoisin/Projects/PythonProjects/gesture-control/data/hand_detect_model/test_img.png")
        return res
