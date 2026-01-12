# The present code is provided only for the purpose of the reviewing process
# of TMLR. Any other usage, copy, edit, distribution
# or code re-use is strictly prohibited.
# Copyright 2023, to the authors of this TMLR submission.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
