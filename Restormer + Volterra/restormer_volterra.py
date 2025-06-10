# Volterra 레이어가 attention과 FFN 양쪽에 모두 삽입된 구조
# E:\MRVNet2D\Restormer + Volterra\restormer_volterra.py
# restormer_volterra.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from volterra_layer import VolterraLayer2D  # 사용자 구현 기반

# ✅ LayerNorm 정의
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))  # [1, C, 1, 1]

    def forward(self, x):
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        x = x / torch.sqrt(var + 1e-5)
        return self.weight * x


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))  # [1, C, 1, 1]
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))   # [1, C, 1, 1]

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return self.weight * x + self.bias


def LayerNorm(normalized_shape, bias=False):
    return WithBiasLayerNorm(normalized_shape) if bias else BiasFreeLayerNorm(normalized_shape)

# ✅ GDFN
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

# ✅ MDTA
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
        out = self.project_out(out)
        return out

# ✅ Transformer Block with Volterra
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

# ✅ Encoder / Decoder
class Encoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])

    def forward(self, x):
        return self.body(x)

class Decoder(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])

    def forward(self, x):
        return self.body(x)

# ✅ Down / Up sample
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

# ✅ 전체 RestormerVolterra 모델
class RestormerVolterra(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=48, num_blocks=[4,6,6,8],
                 num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias', volterra_rank=4):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.encoder1 = Encoder(dim, num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.down1 = Downsample(dim)

        self.encoder2 = Encoder(dim*2, num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.down2 = Downsample(dim*2)

        self.encoder3 = Encoder(dim*4, num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)
        self.down3 = Downsample(dim*4)

        self.latent = Encoder(dim*8, num_blocks[3], num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.up3 = Upsample(dim*8)
        self.decoder3 = Decoder(dim*4, num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.up2 = Upsample(dim*4)
        self.decoder2 = Decoder(dim*2, num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.up1 = Upsample(dim*2)
        self.decoder1 = Decoder(dim, num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.refinement = Encoder(dim, num_refinement_blocks, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, volterra_rank=volterra_rank)

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        print("Input:", x.shape)
        x1 = self.patch_embed(x)
        print("After Patch Embedding:", x1.shape)

        x2 = self.encoder1(x1)
        print("After Encoder1:", x2.shape)

        x3 = self.encoder2(self.down1(x2))
        print("After Encoder2:", x3.shape)

        x4 = self.encoder3(self.down2(x3))
        print("After Encoder3:", x4.shape)

        x5 = self.latent(self.down3(x4))
        print("After Latent:", x5.shape)

        x6 = self.decoder3(self.up3(x5) + x4)
        print("After Decoder3:", x6.shape)

        x7 = self.decoder2(self.up2(x6) + x3)
        print("After Decoder2:", x7.shape)

        x8 = self.decoder1(self.up1(x7) + x2)
        print("After Decoder1:", x8.shape)

        x9 = self.refinement(x8)
        print("After Refinement:", x9.shape)

        out = self.output(x9 + x1)
        print("Final Output:", out.shape)

        return out

if __name__ == '__main__':
    model = RestormerVolterra()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
