import os
import pandas as pd
from glob import glob
from pathlib import Path


def build_manifest(train_annotation: str, test_dir: str, test_annotation: str,
                   train_manifest: str, test_manifest: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    # train
    train_frames: list[pd.DataFrame] = []
    train_annotation_files = sorted(
        glob(train_annotation)
    )

    if not train_annotation_files:
        raise FileNotFoundError(
            f"No train annotation files found: "
            f"{train_annotation}"
        )

    for annotation_file in train_annotation_files:
        annotation_path = Path(annotation_file)

        df = pd.read_csv(annotation_path, sep=";")
        df["img_path"] = df["Filename"].apply(lambda filename: str(annotation_path.parent / filename))
        train_frames.append(df[["img_path", "ClassId"]])  

    train_img_labels = pd.concat(train_frames,ignore_index=True,)
        
    # test
    test_dir_path = Path(test_dir)
    test_img_labels = pd.read_csv(test_annotation, sep=";",)

    test_img_labels["img_path"] = \
        (test_img_labels["Filename"].apply(lambda filename: str(test_dir_path / filename)))

    test_img_labels = test_img_labels[["img_path", "ClassId"]].copy()

    # create output dir
    train_manifest_path = Path(train_manifest)
    test_manifest_path = Path(test_manifest)

    train_manifest_path.parent.mkdir(parents=True,exist_ok=True,)
    test_manifest_path.parent.mkdir(parents=True,exist_ok=True,)
        
    # test
    df = pd.read_csv(test_annotation, sep=";")
    df["img_path"] = df["Filename"].apply(lambda x: os.path.join(test_dir, x))
    df = df[["img_path", "ClassId"]]
    test_img_labels = df.copy()

    # output
    train_img_labels.to_csv(train_manifest, index=False)
    test_img_labels.to_csv(test_manifest, index=False)

    return train_img_labels, test_img_labels


if __name__ == "__main__":
    train_df, test_df = build_manifest(
        train_annotation="data/raw/GTSRB/Final_Training/Images/*/*.csv",
        test_dir="data/raw/GTSRB/Final_Test/Images",
        test_annotation="data/raw/GTSRB/Final_Test/GT-final_test.csv",
        train_manifest="data/processed/train_manifest.csv",
        test_manifest="data/processed/test_manifest.csv"
    )
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
  