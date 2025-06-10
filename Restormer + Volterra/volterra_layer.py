# E:\MRVNet2D\Restormer + Volterra\volterra_layer.py
# volterra_layer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

def circular_shift(x, shift_x, shift_y):
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=2, use_lossless=False):
        super().__init__()
        self.use_lossless = use_lossless
        self.rank = rank

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

        if use_lossless:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            self.shifts = self._generate_shifts(kernel_size)
        else:
            self.W2a = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])
            self.W2b = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])

    def _generate_shifts(self, k):
        P = k // 2
        shifts = []
        for s1 in range(-P, P + 1):
            for s2 in range(-P, P + 1):
                if s1 == 0 and s2 == 0: continue
                if (s1, s2) < (0, 0): continue
                shifts.append((s1, s2))
        return shifts

    def forward(self, x):
        linear_term = self.conv1(x)
        quadratic_term = 0

        if self.use_lossless:
            for s1, s2 in self.shifts:
                x_shifted = circular_shift(x, s1, s2)
                prod = x * x_shifted
                prod = torch.clamp(prod, min=-1.0, max=1.0)
                quadratic_term += self.conv2(prod)
        else:
            for a, b in zip(self.W2a, self.W2b):
                qa = torch.clamp(a(x), min=-1.0, max=1.0)
                qb = torch.clamp(b(x), min=-1.0, max=1.0)
                quadratic_term += qa * qb

        out = linear_term + quadratic_term
        return out
