import torch
import torch.nn as nn
from numpy.random import beta
from .misc import onehot


class MixUp(nn.Module):
    def __init__(self, batch_dim=0):
        super(MixUp, self).__init__()
        self.batch_dim = batch_dim
        self.reset()

    def reset(self):
        self.enabled = False
        self.mix_values = None
        self.mix_index = None

    def mix(self, x1, x2):
        if not torch.is_tensor(self.mix_values):  # scalar
            return x2.lerp(x1, self.mix_values)
        else:
            view = [1] * int(x1.dim())
            view[self.batch_dim] = -1
            mix_val = self.mix_values.to(device=x1.device).view(*view)
            return mix_val * x1 + (1.-mix_val) * x2

    def sample(self, alpha, batch_size, sample_batch=False):
        self.mix_index = torch.randperm(batch_size)
        if sample_batch:
            values = beta(alpha, alpha, size=batch_size)
            self.mix_values = torch.tensor(values, dtype=torch.float)
        else:
            self.mix_values = torch.tensor([beta(alpha, alpha)],
                                           dtype=torch.float)

    def mix_target(self, y, n_class):
        if not self.training or \
            self.mix_values is None or\
                self.mix_values is None:
            return y
        y = onehot(y, n_class).to(dtype=torch.float)
        idx = self.mix_index.to(device=y.device)
        y_mix = y.index_select(self.batch_dim, idx)
        return self.mix(y, y_mix)

    def forward(self, x):
        if not self.training or \
            self.mix_values is None or\
                self.mix_values is None:
            return x
        idx = self.mix_index.to(device=x.device)
        x_mix = x.index_select(self.batch_dim, idx)
        return self.mix(x, x_mix)
