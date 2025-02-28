import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension    # 128
    self.w = torch.nn.Linear(1, dimension) # 1->128

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = torch.log(t + 1)
    t = t.unsqueeze(dim=1)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output
