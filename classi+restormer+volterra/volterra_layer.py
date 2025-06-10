# volterra_layer.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ✅ 원형 순환 시프트 (Lossless Volterra에서 사용)
def circular_shift(x, shift_x, shift_y):
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))


# ✅ Volterra 2차 계층
class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=2, use_lossless=False):
        """
        :param in_channels: 입력 채널 수
        :param out_channels: 출력 채널 수
        :param kernel_size: 기본 컨볼루션 커널 크기 (default: 3)
        :param rank: low-rank approximation 차수 (Lossy 모드에서만 사용)
        :param use_lossless: Lossless Volterra approximation 사용 여부
        """
        super().__init__()
        self.use_lossless = use_lossless
        self.rank = rank

        # 1차 선형 항 (기본 Conv)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        if use_lossless:
            # 2차 항 - 동일한 conv를 곱의 결과에 적용
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            self.shifts = self._generate_shifts(kernel_size)
        else:
            # 2차 항 - 저랭크 근사 방식 (W2a * W2b)
            self.W2a = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])
            self.W2b = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])

    def _generate_shifts(self, k):
        """Lossless 모드에서 사용할 시프트 쌍 생성"""
        P = k // 2
        shifts = []
        for dy in range(-P, P + 1):
            for dx in range(-P, P + 1):
                if dy == 0 and dx == 0:
                    continue
                if (dy, dx) < (0, 0):  # 대칭 제거
                    continue
                shifts.append((dx, dy))
        return shifts

    def forward(self, x):
        # 1차 항
        linear_term = self.conv1(x)
        quadratic_term = 0

        if self.use_lossless:
            # 완전 2차 항 - shift된 이미지와의 곱 사용
            for dx, dy in self.shifts:
                x_shifted = circular_shift(x, dx, dy)
                prod = x * x_shifted
                prod = torch.clamp(prod, min=-1.0, max=1.0)
                quadratic_term += self.conv2(prod)
        else:
            # 저랭크 2차 항 (lossy)
            for conv_a, conv_b in zip(self.W2a, self.W2b):
                qa = torch.clamp(conv_a(x), min=-1.0, max=1.0)
                qb = torch.clamp(conv_b(x), min=-1.0, max=1.0)
                quadratic_term += qa * qb

        return linear_term + quadratic_term
