
import torch
import pytest

from PIL import Image
from torchvision.transforms import v2
from src.preprocess.transforms import build_transforms

def test_build_transforms():

    # arrange
    image_input = Image.new(
        mode="RGB",
        size=(20,10),
        color=(255,255,255),   
    )

    transform = build_transforms()
    # act
    image_output = transform(image_input)

    # assert
    assert image_output.shape == (3, 32, 32)
    assert image_output.dtype == torch.float32
    assert image_output.min().item() == 1.0
    assert image_output.max().item() == 1.0



