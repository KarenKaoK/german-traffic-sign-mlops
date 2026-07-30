import pandas as pd
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        transform=None,
    ):

        self.manifest = pd.read_csv(manifest_path)
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        sample = self.manifest.iloc[index]

        img_path = sample["img_path"]
        label = int(sample["ClassId"])

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":

    dataset = GTSRBDataset(
        manifest_path="data/processed/train_manifest.csv",
    )

    print("len dataset", len(dataset))

    image, label = dataset[0]

    print("image type", type(image))
    print("Image size:", image.size)
    print("Label:", label)
