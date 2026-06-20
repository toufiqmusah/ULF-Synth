import torch
from torch import nn, Tensor
import numpy as np


class myMAE(nn.L1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)
