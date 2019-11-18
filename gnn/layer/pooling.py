"""
Pooling leayers
"""

from torch import nn
import torch


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
