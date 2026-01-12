# The present code is provided only for the purpose of the reviewing process
# of TMLR. Any other usage, copy, edit, distribution
# or code re-use is strictly prohibited.
# Copyright 2023, to the authors of this TMLR submission.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Normalization modules."""

import typing as tp

import einops
import torch
from torch import nn


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return
