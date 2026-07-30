import pandas as pd
from PIL import Image

from pathlib import Path
from src.data.dataset import GTSRBDataset


def test_dataset_length(
    tmp_path: Path,
):
    # Arrange
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "img_path": ["image_1.ppm"],
            "ClassId": [0],
        }
    ).to_csv(
        manifest_path,
        index=False,
    )

    # Act
    dataset = GTSRBDataset(
        manifest_path=manifest_path,
    )

    # Assert
    assert len(dataset) == 1


def test_dataset_returns_image_and_label(
    tmp_path: Path,
):

    # Arrange
    manifest_path = tmp_path / "manifest.csv"
    image_path = tmp_path / "image_1.ppm"
    pd.DataFrame(
        {
            "img_path": [str(image_path)],
            "ClassId": [1],
        }
    ).to_csv(
        manifest_path,
        index=False,
    )

    test_image = Image.new(
        mode="L",
        size=(20, 10),
    )

    test_image.save(image_path)

    # Act
    dataset = GTSRBDataset(
        manifest_path=manifest_path,
    )

    image, label = dataset[0]

    # Assert
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (20, 10)
    assert label == 1
    assert isinstance(label, int)


def test_dataset_applies_transform(
    tmp_path: Path,
):

    # Arrange
    manifest_path = tmp_path / "manifest.csv"
    image_path = tmp_path / "image_1.ppm"
    pd.DataFrame(
        {
            "img_path": [str(image_path)],
            "ClassId": [1],
        }
    ).to_csv(
        manifest_path,
        index=False,
    )

    test_image = Image.new(
        mode="RGB",
        size=(20, 10),
    )

    test_image.save(image_path)

    # fake transform
    def fake_transform(image: Image.Image):
        return image.resize((5, 5))

    # Act
    dataset = GTSRBDataset(manifest_path=manifest_path, transform=fake_transform)

    image, label = dataset[0]

    # Assert
    assert image.size == (5, 5)
    assert label == 1
