"""NB. We emprically got a massive speedup by doing these transforms on the GPU during
the training step, as opposed to in the DataLoader as standard."""

import torch
import torch.nn as nn
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

normalise_2d = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def get_transforms_for_eval() -> nn.Module:
    """Evaluation (val/test) transforms. Input to the pipeline is a
    RGB uint8 Tensor, output is a normalised float Tensor.

    Returns:
        nn.Module: Sequential set of transforms. Call this on the images during the
            training step on GPU.
    """
    transform_list = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        normalise_2d,
    ]

    return nn.Sequential(*transform_list)


def get_transforms_for_train(augment: bool = True) -> nn.Module:
    """Transforms for training data. If augment is True, apply autoaugment policy.

    Input to the pipeline is a RGB uint8 Tensor, output is a normalised float Tensor.

    Args:
        augment (bool, optional): Whether to augment data. If False, simply return
            the eval transforms. Defaults to True.

    Returns:
        nn.Module: Sequential set of transforms. Call this on the images during the
            training step on GPU.
    """
    if not augment:
        return get_transforms_for_eval()

    transform_list = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(15),
        normalise_2d,
    ]

    return nn.Sequential(*transform_list)
