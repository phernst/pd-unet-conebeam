from typing import NamedTuple

import torch


class SpaceNormalization(NamedTuple):
    sino: float
    img: float


class ZNorm(object):
    def __init__(self, tensor: torch.Tensor):
        self.mean = tensor.mean(dim=(2, 3), keepdims=True)
        self.std = tensor.std(dim=(2, 3), keepdims=True)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean)/self.std

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean
