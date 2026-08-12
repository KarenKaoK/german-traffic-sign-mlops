import torch
from torch.utils.data import TensorDataset
from src.data.split_dataset import split_train_val


def test_split_train_val_correct_sizes():

    # arrange
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 43, (100,))
    dataset = TensorDataset(images, labels)

    val_ratio = 0.1
    seed = 10

    # act
    train_dataset, val_dataset = split_train_val(
        dataset=dataset, val_ratio=val_ratio, seed=seed
    )

    # assert
    assert len(train_dataset) == 90
    assert len(val_dataset) == 10


def test_split_train_val_same_seed_same_split():

    # arrange
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 43, (100,))
    dataset = TensorDataset(images, labels)

    val_ratio = 0.1
    seed = 10

    # act
    train_dataset_1, val_dataset_1 = split_train_val(
        dataset=dataset, val_ratio=val_ratio, seed=seed
    )

    train_dataset_2, val_dataset_2 = split_train_val(
        dataset=dataset, val_ratio=val_ratio, seed=seed
    )

    # assert
    assert train_dataset_1.indices == train_dataset_2.indices
    assert val_dataset_1.indices == val_dataset_2.indices
