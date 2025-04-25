from abc import ABC, abstractmethod
import torch.nn as nn


class BaseTransformer(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        """ x: Tensor of shape (B, C, T, H, W) """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        """ Instantiate model from config dict or path """
        pass
