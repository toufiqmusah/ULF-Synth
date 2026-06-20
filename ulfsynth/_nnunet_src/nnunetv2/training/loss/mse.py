import torch
from torch import nn, Tensor
import numpy as np


class myMSE(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target) 
