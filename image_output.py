import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

# ---------------- LayerNorm ---------------- #
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(var + 1e-5)
        return self.weight * x

class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return self.weight * x + self.bias

def LayerNorm(normalized_shape, bias=False):
    return WithBiasLayerNorm(normalized_shape) if bias else BiasFreeLayerNorm(normalized_shape)


# ---------------- GDFN ---------------- #
class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ---------------- MDTA ---------------- #
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


# ---------------- Volterra ---------------- #
class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=2, use_lossless=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.W2a = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            for _ in range(rank)
        ])
        self.W2b = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            for _ in range(rank)
        ])

    def forward(self, x):
        linear_term = self.conv1(x)
        quadratic_term = 0
        for a, b in zip(self.W2a, self.W2b):
            qa = torch.clamp(a(x), min=-1.0, max=1.0)
            qb = torch.clamp(b(x), min=-1.0, max=1.0)
            quadratic_term += qa * qb
        return linear_term + quadratic_term


# ---------------- Transformer Block (VET) ---------------- #
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, volterra_rank):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=bias) if LayerNorm_type == 'WithBias' else BiasFreeLayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.volterra1 = VolterraLayer2D(dim, dim, rank=volterra_rank)
        self.norm2 = LayerNorm(dim, bias=bias) if LayerNorm_type == 'WithBias' else BiasFreeLayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)
        self.volterra2 = VolterraLayer2D(dim, dim, rank=volterra_rank)

    def forward(self, x):
        x = x + self.volterra1(self.attn(self.norm1(x)))
        x = x + self.volterra2(self.ffn(self.norm2(x)))
        return x


# ---------------- Encoder ---------------- #
class Encoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])

    def forward(self, x):
        return self.body(x)


# ---------------- Restormer + Volterra (간단화 버전) ---------------- #
class RestormerVolterra(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=48,
                 num_blocks=1, heads=1, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias', volterra_rank=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.encoder = Encoder(dim, num_blocks, num_heads=heads,
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.patch_embed(x)
        x2 = self.encoder(x1)
        out = self.output(x2 + x1)
        return out


# ---------------- Feature Hook (Overlay) ---------------- #
SAVE_DIR = "feature_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

# 원본 이미지 저장 (후처리용)
ORIG_IMG = None

def save_feature_map(name):
    def hook_fn(module, input, output):
        global ORIG_IMG
        feat = output.detach().cpu().numpy()[0]   # (C,H,W)
        feat_mean = feat.mean(axis=0)             # 채널 평균 (H,W)

        # normalize 0-255
        feat_norm = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
        feat_norm = (feat_norm * 255).astype(np.uint8)

        # heatmap 생성
        heatmap = cv2.applyColorMap(feat_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 원본 크기에 맞추기
        heatmap = cv2.resize(heatmap, (ORIG_IMG.shape[1], ORIG_IMG.shape[0]))

        # overlay
        overlay = cv2.addWeighted(ORIG_IMG, 0.6, heatmap, 0.4, 0)

        # 저장
        out_path = os.path.join(SAVE_DIR, f"{name}.png")
        Image.fromarray(overlay).save(out_path)
    return hook_fn


# ---------------- 실행 ---------------- #
if __name__ == "__main__":
    model = RestormerVolterra()

    # TransformerBlock 선택 (첫 블록)
    block = model.encoder.body[0]

    # Hook 등록
    block.norm1.register_forward_hook(save_feature_map("01_norm1"))
    block.attn.register_forward_hook(save_feature_map("02_mdta"))
    block.volterra1.register_forward_hook(save_feature_map("03_volterra1"))
    block.norm2.register_forward_hook(save_feature_map("04_norm2"))
    block.ffn.register_forward_hook(save_feature_map("05_gdfn"))
    block.volterra2.register_forward_hook(save_feature_map("06_volterra2"))

    # 입력 이미지 불러오기
    img_path = r"E:\restormer+volterra\data\KADID10K\images\I26_03_05.png"
    img = Image.open(img_path).convert("RGB")
    ORIG_IMG = np.array(img.resize((256,256)))  # overlay용 numpy

    transform = T.Compose([
        T.Resize((256,256)),
        T.ToTensor()
    ])
    inp = transform(img).unsqueeze(0)  # (1,3,256,256)

    # 모델 실행
    with torch.no_grad():
        out = model(inp)

    print("Output shape:", out.shape)
    print(f"중간 feature overlay 저장 완료 → {SAVE_DIR}")
