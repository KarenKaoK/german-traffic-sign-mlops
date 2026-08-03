import torch
from torch.utils.data import TensorDataset

from src.data.dataloader import build_dataloader


def test_build_dataloader():

    # arrange
    test_images = torch.randn(4, 3, 32, 32)
    test_labels = torch.tensor([0, 1, 2, 3])

    dataset = TensorDataset(test_images, test_labels)

    test_batch_size = 2

    # act
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=test_batch_size,
        shuffle=True,
    )

    images, labels = next(iter(dataloader))

    # assert
    assert images.shape == (2, 3, 32, 32)
    assert labels.shape == (2,)
