import torch
from torch.utils.data import Dataset, random_split


def split_train_val(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
):

    total_size = len(dataset)

    val_size = round(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset
