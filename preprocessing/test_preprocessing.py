import dataclasses
import os
from typing import Tuple

import pandas as pd
import pytest
import torchvision
import torch


@dataclasses.dataclass
class DatasetSplits:
    train_split: pd.DataFrame
    val_split: pd.DataFrame
    test_split: pd.DataFrame

    SPLIT_DIR: str
    DATA_DIR: str
    SENSITIVE_ATTRIBUTES: Tuple[str, ...]


@pytest.fixture(
    name="dataset_splits", params=["ham10000", "chexpert", "mimic", "papila", "fitzpatrick17k"]
)
def _dataset_splits(request) -> DatasetSplits:
    if request.param == "ham10000":
        SPLIT_DIR = "splits/ham10000/"
        DATA_DIR = "/mnt/HAM10000/"
        SENSITIVE_ATTRIBUTES = ("Age", "Sex")

    elif request.param == "chexpert":
        SPLIT_DIR = "splits/chexpert/"
        DATA_DIR = "/mnt/chest_xray/CheXpert-v1.0/"
        SENSITIVE_ATTRIBUTES = ("Age", "Race", "Sex")

    elif request.param == "mimic":
        SPLIT_DIR = "splits/mimic/"
        DATA_DIR = "/mnt/chest_xray/mimic-cxr-jpg-224/data/"
        SENSITIVE_ATTRIBUTES = ("Age", "Race", "Sex")

    elif request.param == "papila":
        SPLIT_DIR = "splits/papila/"
        DATA_DIR = "/mnt/PAPILA/"
        SENSITIVE_ATTRIBUTES = ("Age", "Sex")

    elif request.param == "fitzpatrick17k":
        SPLIT_DIR = "splits/fitzpatrick17k/"
        DATA_DIR = "/mnt/fitzpatrick17k/"
        SENSITIVE_ATTRIBUTES = ("SkinType",)

    else:
        raise ValueError(f"Unknown dataset {request.param}")

    train_split = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_split = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test_split = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    return DatasetSplits(
        train_split,
        val_split,
        test_split,
        SPLIT_DIR,
        DATA_DIR,
        SENSITIVE_ATTRIBUTES,
    )


def test_unique_imgs(dataset_splits: DatasetSplits):
    """Check that all images in each split are unique"""
    train_split, val_split, test_split = (
        dataset_splits.train_split,
        dataset_splits.val_split,
        dataset_splits.test_split,
    )
    assert len(train_split["Path"].unique()) == len(train_split)
    assert len(val_split["Path"].unique()) == len(val_split)
    assert len(test_split["Path"].unique()) == len(test_split)


def test_disjoint_splits(dataset_splits: DatasetSplits):
    """Check that the splits are disjoint"""
    train_split, val_split, test_split = (
        dataset_splits.train_split,
        dataset_splits.val_split,
        dataset_splits.test_split,
    )
    assert (
        len(
            set(train_split["Path"].unique()).intersection(
                set(val_split["Path"].unique())
            )
        )
        == 0
    )
    assert (
        len(
            set(train_split["Path"].unique()).intersection(
                set(test_split["Path"].unique())
            )
        )
        == 0
    )
    assert (
        len(
            set(val_split["Path"].unique()).intersection(
                set(test_split["Path"].unique())
            )
        )
        == 0
    )


def test_img_dtype(dataset_splits: DatasetSplits):
    """Check that a random image in a split can be read into a uint8 tensor"""
    train_split, val_split, test_split = (
        dataset_splits.train_split,
        dataset_splits.val_split,
        dataset_splits.test_split,
    )
    DATA_DIR = dataset_splits.DATA_DIR

    for df in [train_split, val_split, test_split]:
        img_path = os.path.join(DATA_DIR, df.iloc[0]["Path"])
        img = torchvision.io.read_image(img_path)
        assert img.dtype == torch.uint8


def test_sensitive_attributes(dataset_splits: DatasetSplits):
    """Check that the sensitive attributes are present and are integer type."""

    train_split, val_split, test_split = (
        dataset_splits.train_split,
        dataset_splits.val_split,
        dataset_splits.test_split,
    )
    SENSITIVE_ATTRIBUTES = dataset_splits.SENSITIVE_ATTRIBUTES

    for df in [train_split, val_split, test_split]:
        for attr in SENSITIVE_ATTRIBUTES:
            assert attr in df.columns
            assert df[attr].dtype == "int64"
            # check that every column has at least one non-zero value
            assert (df[attr] != 0).any()
