import os
from enum import Enum
from typing import Literal, Mapping, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypeGuard


class SensitiveAttribute(Enum):
    """All datasets use one or more of these attributes and are binarized in the
    following way. We expect the csv files to have a column with the same name as
    the Key field in the mapping below."""

    sex = {0: "Male", 1: "Female", "Key": "Sex"}
    age = {0: "0-60", 1: "60+", "Key": "Age"}
    race = {0: "White", 1: "Non-White", "Key": "Race"}
    skintype = {0: "Types 1-3", 1: "Types 4-6", "Key": "SkinType"}


# NB, we also expect the csv file to have a column named "Path" with the paths from
# data_dir to images and a column named "binaryLabel" with the disease label


class AvailableDataset(Enum):
    """Register datasets by adding them and their available sensitive attributes here."""

    chexpert = [SensitiveAttribute.sex, SensitiveAttribute.age, SensitiveAttribute.race]
    mimic = [SensitiveAttribute.sex, SensitiveAttribute.age, SensitiveAttribute.race]
    ham10000 = [SensitiveAttribute.sex, SensitiveAttribute.age]
    papila = [SensitiveAttribute.sex, SensitiveAttribute.age]
    fitzpatrick17k = [SensitiveAttribute.skintype]


def is_valid_sensitive_attribute(
    a: SensitiveAttribute, ds: AvailableDataset
) -> TypeGuard[SensitiveAttribute]:
    if a not in ds.value:
        raise ValueError(f"Invalid sensitive attribute {a} for dataset {ds}")
    return True


class FairnessDataset(Dataset):
    def __init__(
        self,
        split_dir: str,
        data_dir: str,
        split: Literal["train", "val", "test"],
        sensitive_attribute: SensitiveAttribute,
        dataset_name: AvailableDataset,
        label_noise: float = 0.0,
    ) -> None:
        super().__init__()
        assert is_valid_sensitive_attribute(sensitive_attribute, dataset_name)

        split_path = os.path.join(split_dir, f"{split}.csv")
        self.split_df = pd.read_csv(split_path)
        self.data_dir = data_dir
        self.sensitive_attribute = sensitive_attribute

        self.label_noise = label_noise

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.data_dir, self.split_df.iloc[idx]["Path"])
        img = torchvision.io.read_image(
            img_path, mode=torchvision.io.image.ImageReadMode.RGB
        )

        label = self.split_df.iloc[idx]["binaryLabel"]
        label = torch.tensor(label, dtype=torch.long)
        subgroup = self.split_df.iloc[idx][self.sensitive_attribute.value["Key"]]
        subgroup = torch.tensor(subgroup, dtype=torch.long)

        # label noise is applied by mislabelling the positive class in subgroup 1
        if self.label_noise > 0.0 and subgroup == 1 and label == 1:
            if torch.rand(1) < self.label_noise:
                label = torch.tensor(0, dtype=torch.long)

        return img, label, subgroup

    def __len__(self):
        return len(self.split_df)


class FairnessDataModule(pl.LightningDataModule):
    def __init__(
        self,
        split_dir: str,
        data_dir: str,
        sensitive_attribute: str,
        dataset_name: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        label_noise: float = 0.0,
    ) -> None:
        """
        Args:
            split_dir (str): Path to directory containing train.csv, val.csv, and
                test.csv for the given dataset.
            data_dir (str): Path to directory containing the dataset images.
            sensitive_attribute (str): Case-insensitive name of the sensitive attribute
                to use. It will be converted into a SensitiveAttribute enum.
            dataset_name (str): Case-insensitive name of the dataset to use. It will be
                converted into an AvailableDataset enum.
            batch_size (int): Batch size to use.
            num_workers (int): Number of workers to use for data loading.
            pin_memory (bool): Whether to pin memory for data loading.
            label_noise (float): Probability of mislabelling the positive class in
                subgroup 1. Applied on training and validation set but never on test
                set. Defaults to 0.0.
        """
        super().__init__()
        self.split_dir = split_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.sensitive_attribute = SensitiveAttribute[sensitive_attribute.lower()]
        self.dataset_name = AvailableDataset[dataset_name.lower()]

        self.label_noise = label_noise

    def setup(self, stage: str) -> None:
        assert stage in ("fit", "test")
        assert is_valid_sensitive_attribute(self.sensitive_attribute, self.dataset_name)

        if stage == "fit":
            # train and val datasets have label noise
            self.train_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "train",
                self.sensitive_attribute,
                self.dataset_name,
                self.label_noise,
            )
            self.val_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "val",
                self.sensitive_attribute,
                self.dataset_name,
                self.label_noise,
            )
        elif stage == "test":
            self.test_dataset = FairnessDataset(
                self.split_dir,
                self.data_dir,
                "test",
                self.sensitive_attribute,
                self.dataset_name,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def get_subgroup_names(self) -> Mapping[int, str]:
        return self.sensitive_attribute.value
