from typing import List, Literal, Mapping, Tuple

import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from models import metrics
from models import transforms


class Baseline(pl.LightningModule):
    """Baseline (ERM) classifier for Binary Classification, with 2 subgroups.
    Can optionally be trained to predict the sensitive attribute instead of the disease.
    """

    def __init__(
        self,
        lr: float = 3e-4,
        predict_attr: bool = False,
        backbone: nn.Module = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        ),
    ) -> None:
        """
        Args:

            lr (float, optional): Learning rate. Defaults to 3e-4.
            predict_attr (bool, optional): Whether to predict the sensitive attribute
                instead of the disease label. When False, the disease prediction
                metrics are stratified by subgroup. Defaults to False.
            backbone (nn.Module, optional): Backbone model. Defaults to the torchvision
                ResNet50 implementation.
        """
        super().__init__()
        self.lr = lr
        self.predict_attr = predict_attr

        self.backbone = backbone
        self.prediction_head = nn.Linear(1000, 1)  # always binary classification

        self.train_transform = transforms.get_transforms_for_train()
        self.eval_transform = transforms.get_transforms_for_eval()

        # whether to perform the supervised prediction layer information test
        self.SPLIT = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.prediction_head(x)  # note: no sigmoid here
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        transform: nn.Module,
    ) -> Mapping[str, torch.Tensor]:
        imgs, labels, attributes = batch
        imgs = transform(imgs)
        logits = self(imgs)

        if self.predict_attr:
            y = attributes
        else:
            y = labels

        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y.float())
        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
            "attributes": attributes,
        }

    def _epoch_end(
        self,
        step_outputs: List[Mapping[str, torch.Tensor]],
        mode: Literal["train", "val", "test"],
    ):
        labels = torch.cat([x["labels"] for x in step_outputs]).cpu().detach().numpy()
        attributes = (
            torch.cat([x["attributes"] for x in step_outputs]).cpu().detach().numpy()
        )

        logits = torch.cat([x["logits"] for x in step_outputs])
        scores = torch.sigmoid(logits).squeeze().cpu().detach().numpy()

        loss = torch.stack([x["loss"] for x in step_outputs]).mean().item()

        if self.predict_attr:
            metrics_dict = metrics.compute_overall_metrics(
                scores=scores, labels=attributes
            )

        else:
            metrics_dict = metrics.compute_fairness_metrics(
                scores=scores, labels=labels, attributes=attributes
            )

        metrics_dict["loss"] = loss

        # modify the keys to include the mode, also convert back to torch.Tensor
        torch_metrics_dict_with_mode = {
            f"{'split_' if self.SPLIT else ''}{mode}/{k}": v
            for k, v in metrics_dict.items()
        }
        self.log_dict(torch_metrics_dict_with_mode, on_epoch=True, prog_bar=True)

    def set_SPLIT(self):
        """SPLIT is the supervised prediction layer information test. It involves
        freezing the backbone and re-training the prediction layer to see if the
        representations encode subgroup information. Call this method, then train
        and test again to obtain SPLIT metrics."""

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.predict_attr = True
        self.SPLIT = True

    def training_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, self.train_transform)

    def training_epoch_end(self, step_outputs: List[Mapping[str, torch.Tensor]]):
        self._epoch_end(step_outputs, "train")

    def validation_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, self.eval_transform)

    def validation_epoch_end(self, step_outputs: List[Mapping[str, torch.Tensor]]):
        if self.trainer.state.stage == "sanity_check":
            return
        self._epoch_end(step_outputs, "val")

    def test_step(self, batch, batch_idx):
        del batch_idx
        return self._step(batch, self.eval_transform)

    def test_epoch_end(self, step_outputs: List[Mapping[str, torch.Tensor]]):
        self._epoch_end(step_outputs, "test")
