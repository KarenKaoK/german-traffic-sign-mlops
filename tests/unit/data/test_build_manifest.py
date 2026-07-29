import pandas as pd
from pathlib import Path

from src.data.build_manifest import build_manifest

def test_build_manifest_creates_train_and_test_manifest(tmp_path: Path,):

    # Arrange
    train_dir = tmp_path / "train" /"images"
    test_dir = tmp_path / "test" 
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_image_path = train_dir / "00001" / "00001_00000.ppm"
    train_image_path.parent.mkdir(parents=True, exist_ok=True)
    train_image_path.touch()


    test_root_dir = tmp_path / "test"
    test_image_dir = test_root_dir / "images"
    test_image_dir.mkdir(parents=True, exist_ok=True)

    test_image_path = test_image_dir / "00001_00000.ppm"
    test_image_path.touch()


    train_annotation_file = train_dir / "00001" / "GT-00001.csv"
    pd.DataFrame(
        {
            "Filename": ["00001_00000.ppm"],
            "ClassId": [0],
        }
    ).to_csv(
        train_annotation_file,
        sep=";",
        index=False,
    )

    test_annotation_file = test_root_dir / "GT-final_test.csv"
    pd.DataFrame(
        {
            "Filename": ["00001_00000.ppm"],
            "ClassId": [0],
        }
    ).to_csv(
        test_annotation_file,
        sep=";",
        index=False,        
    )

    train_manifest = tmp_path / "processed" / "train_manifest.csv"
    test_manifest = tmp_path / "processed" / "test_manifest.csv"

    # Act
    train_df, test_df = build_manifest(
        train_annotation=str(train_dir / "*" / "GT-*.csv"),
        test_dir=str(test_image_dir),
        test_annotation=str(test_annotation_file),
        train_manifest=str(train_manifest),
        test_manifest=str(test_manifest),
    )

    # Assert
    assert len(train_df) == 1
    assert len(test_df) == 1

    assert train_manifest.exists()
    assert test_manifest.exists()

    assert train_df.iloc[0]["ClassId"] == 0
    assert test_df.iloc[0]["ClassId"] == 0

    assert train_df.iloc[0]["img_path"] == str(train_image_path)
    assert test_df.iloc[0]["img_path"] == str(test_image_path)