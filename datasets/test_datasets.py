import os
from typing import Tuple

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

import models.transforms as transforms

pl.seed_everything(100)

# register datasets to test by adding them to this list,
# the name registered here must match the name of the config file without the extension
DATASETS = ["ham10000", "chexpert", "mimic", "papila", "fitzpatrick17k"]


# Here we set up the datamodules with the same parameters as in our config files.
# This allows us to keep our tests in sync with our configuration.
@pytest.mark.skip(reason="Not a test, just a helper function.")
def get_datamodule(ds: str) -> pl.LightningDataModule:
    if ds in DATASETS:
        overrides = [f"datamodule={ds}"]

    else:
        raise ValueError(f"Unknown dataset {ds}")

    with initialize(version_base="1.2", config_path="../configs/", job_name="test_ds"):
        cfg = compose(config_name="config.yaml", overrides=overrides)
        dm = instantiate(cfg.datamodule)
        dm.setup(stage="fit")
        dm.setup(stage="test")
        return dm


@pytest.fixture(name="datamodule", params=DATASETS, scope="module")
def _datamodule(request) -> pl.LightningDataModule:
    return get_datamodule(request.param)


def test_shapes_and_dtypes(datamodule):
    """Test if the shapes and dtypes of train and val batches are correct."""

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    batch_size = datamodule.batch_size  # type: ignore

    for batch, transform in zip(
        [train_batch, val_batch],
        [transforms.get_transforms_for_train(), transforms.get_transforms_for_eval()],
    ):
        imgs, labels, subgroups = batch
        imgs = transform(imgs)

        assert imgs.shape == (batch_size, 3, 224, 224)
        assert labels.shape == (batch_size,)
        assert subgroups.shape == (batch_size,)

        assert imgs.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert subgroups.dtype == torch.int64


def test_visualize_data(datamodule):
    """Visualize the data to make sure it is correct. Shows a 10x2 grid of images: train
    on top and val on bottom. Test always passes, but saves the image to
    test_outputs/test_datasets/<dataset_name>.png"""

    train_batch: Tuple[torch.Tensor, ...] = next(iter(datamodule.train_dataloader()))  # type: ignore
    val_batch: Tuple[torch.Tensor, ...] = next(iter(datamodule.val_dataloader()))  # type: ignore
    test_batch: Tuple[torch.Tensor, ...] = next(iter(datamodule.test_dataloader()))  # type: ignore

    train_batch = (
        transforms.get_transforms_for_train()(train_batch[0]),
        *train_batch[1:],
    )
    val_batch = (transforms.get_transforms_for_eval()(val_batch[0]), *val_batch[1:])
    test_batch = (transforms.get_transforms_for_eval()(test_batch[0]), *test_batch[1:])

    fig, axs = plt.subplots(3, 10, figsize=(25, 10))
    for i in range(10):
        axs[0, i].imshow(train_batch[0][i].permute(1, 2, 0).detach().numpy())  # type: ignore
        axs[1, i].imshow(val_batch[0][i].permute(1, 2, 0))  # type: ignore
        axs[2, i].imshow(test_batch[0][i].permute(1, 2, 0))  # type: ignore

        axs[0, i].set_title(f"Label: {train_batch[1][i].detach().numpy()} \n Subgroup: {train_batch[2][i].detach().numpy()}")  # type: ignore
        axs[1, i].set_title(f"Label: {val_batch[1][i].detach().numpy()} \n Subgroup: {val_batch[2][i].detach().numpy()}")  # type: ignore
        axs[2, i].set_title(f"Label: {test_batch[1][i].detach().numpy()} \n Subgroup: {test_batch[2][i].detach().numpy()}")  # type: ignore

        axs[0, i].axis("off")  # type: ignore
        axs[1, i].axis("off")  # type: ignore
        axs[2, i].axis("off")  # type: ignore

    # label top row as train and bottom row as val
    plt.text(-0.5, 0.5, "Train", ha="center", va="center", transform=axs[0, 0].transAxes)  # type: ignore
    plt.text(-0.5, 0.5, "Val", ha="center", va="center", transform=axs[1, 0].transAxes)  # type: ignore
    plt.text(-0.5, 0.5, "Test", ha="center", va="center", transform=axs[2, 0].transAxes)  # type: ignore

    plt.tight_layout()
    os.makedirs("test_outputs/test_datasets", exist_ok=True)

    # we get the dataset name from by taking the last part of the split_dir path
    plt.savefig(
        f"test_outputs/test_datasets/{os.path.basename(os.path.normpath(datamodule.split_dir))}.png"
    )
    plt.close()
    assert True
