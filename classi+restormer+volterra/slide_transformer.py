import torch
import torch.nn as nn
from timm.layers import trunc_normal_

### Feature Shift → Depthwise Convolution 변환을 적용한 Slide Attention ###
class SlideAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.relative_position_bias_table = None

    def forward(self, x):
        B, N, C = x.shape  
        H = W = int(N ** 0.5)  

        if self.relative_position_bias_table is None or self.relative_position_bias_table.shape[-1] != H * W:
            self.relative_position_bias_table = nn.Parameter(torch.zeros(1, self.num_heads, H * W, H * W))
            trunc_normal_(self.relative_position_bias_table, std=.02)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.relative_position_bias_table.to(attn.device)[:, :, :N, :N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)로 변환
        x = self.depthwise_conv(x)  # Feature Shift 적용 (Depthwise Conv)
        x = x.flatten(2).transpose(1, 2)  # 다시 (B, N, C) 형태로 변환

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


### Deformed Shifting Module 적용 (Predefined Shift + Learnable Shift + Re-parameterization) ###
class DeformedShifting(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.predefined_shift = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)
        self.learnable_shift = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=False)

    def forward(self, x, mode="train"):
        if mode == "train":
            shifted_x1 = self.predefined_shift(x)
            shifted_x2 = self.learnable_shift(x)
            return shifted_x1 + shifted_x2  # 학습 가능한 이동 + 고정 이동을 합쳐서 사용
        else:
            # Inference 시 하나의 경로로 병합하여 연산 최적화 (Re-parameterization)
            merged_weight = self.predefined_shift.weight + self.learnable_shift.weight
            optimized_shift = nn.Conv2d(x.size(1), x.size(1), kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=x.size(1), bias=False)
            optimized_shift.weight = nn.Parameter(merged_weight)
            return optimized_shift(x)


### Slide Transformer 전체 구조 ###
class SlideTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=6, embed_dim=96, num_heads=6, mlp_ratio=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.deformed_shifting = DeformedShifting(embed_dim, kernel_size=3)
        self.attn_layer = SlideAttention(dim=embed_dim, num_heads=num_heads)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=4  # ✅ 4개 블록 사용
        )


        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mode="train"):
        B = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.deformed_shifting(x, mode=mode)  # Feature Shift + Learnable Shift 적용
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) 변환
        x = self.attn_layer(x)  # Slide Attention 적용
        x = self.mlp(x.mean(dim=1))  # MLP
        x = self.global_transformer(x)  # ✅ 여러 개의 Transformer 블록 통과
  # Global Transformer 적용

        return self.head(x)


# ✅ 테스트 실행
if __name__ == "__main__":
    model = SlideTransformer(img_size=224, num_classes=6)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 훈련(Training) 단계
    output_train = model(dummy_input, mode="train")
    print("Training Output Shape:", output_train.shape)

    # 추론(Inference) 단계 (Re-parameterization 적용)
    output_infer = model(dummy_input, mode="infer")
    print("Inference Output Shape:", output_infer.shape)



