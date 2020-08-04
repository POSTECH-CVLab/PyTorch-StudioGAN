# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/linear_classifier.py


from models.model_ops import *

import torch
import torch.nn as nn
import torch.nn.functional as F



class linear_classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(linear_classifier, self).__init__()
        self.linear = linear(in_features=in_channels, out_features=num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out