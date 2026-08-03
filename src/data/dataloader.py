from torch.utils.data import DataLoader

from src.data.dataset import GTSRBDataset
from src.preprocess.transforms import build_transforms


def build_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":

    dataset = GTSRBDataset(
        manifest_path="data/processed/train_manifest.csv",
        transform=build_transforms(),
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
    )

    images, labels = next(iter(dataloader))

    print(images.shape)
    print(labels.shape)
