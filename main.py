import logging

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger  # type: ignore
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from datasets.dataset import FairnessDataModule
import models.baseline


@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get()["runtime"]["output_dir"]  # type: ignore
    pl.seed_everything(cfg.rng, workers=True)

    dm: FairnessDataModule = hydra.utils.instantiate(cfg.datamodule)
    module: models.baseline.Baseline = hydra.utils.instantiate(cfg.model)

    precision = 32
    if cfg.mixed_precision:
        torch.set_float32_matmul_precision("medium")
        precision = 16

    loggers = []
    if cfg.csv_logger:
        csv_logger = CSVLogger(save_dir=output_dir)
        loggers.append(csv_logger)
    if cfg.wandb_logger:
        wandb_logger = WandbLogger(save_dir=output_dir, project=cfg.project_name)
        wandb_logger.watch(module, log="all", log_freq=100)
        loggers.append(wandb_logger)
    if cfg.tb_logger:
        tb_logger = TensorBoardLogger(save_dir=output_dir)  # type: ignore
        loggers.append(tb_logger)

    logging.info(f"Subgroups available in this dataset: {dm.get_subgroup_names()}")

    callbacks = []
    if cfg.early_stopping:
        es = EarlyStopping(
            monitor="val/loss",
            patience=cfg.early_stopping,
            mode="min",
            verbose=True,
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)
    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator="gpu",
        logger=loggers,
        enable_checkpointing=cfg.enable_checkpointing,
        precision=precision,
        default_root_dir=output_dir,
        callbacks=callbacks,
    )
    trainer.fit(module, dm)
    trainer.test(module, datamodule=dm)

    if cfg.split_test and not cfg.model.predict_attr:
        # perform SPLIT test by re-training to predict attribute with frozen backbone
        callbacks = []
        if cfg.early_stopping:
            es = EarlyStopping(
                monitor="split_val/loss",
                patience=cfg.early_stopping,
                mode="min",
                verbose=True,
                check_finite=True,
                strict=True,
            )
            callbacks.append(es)
        split_trainer = pl.Trainer(
            max_epochs=cfg.num_epochs,
            accelerator="gpu",
            logger=loggers,
            enable_checkpointing=False,
            precision=precision,
            default_root_dir=output_dir,
            callbacks=callbacks,
        )
        module.set_SPLIT()
        split_trainer.fit(module, dm)
        split_trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()
