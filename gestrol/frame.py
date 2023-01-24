# external libraries
import torch
from PIL.Image import Image
from torchvision import transforms


class Frame(torch.Tensor):
    """Data model for a single frame."""

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    @classmethod
    def from_pil(cls, pil_image: Image):
        """Create a Frame object from a PIL image."""
        converter = transforms.ToTensor()
        return cls(converter(pil_image))
