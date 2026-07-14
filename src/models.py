"""Model architectures: plain fine-tuned ResNet, self-attention augmented
ResNet, and a dual-input (two-image) variant.
"""

import torch
import torch.nn as nn
from torchvision import models


def _load_backbone(name: str, pretrained: bool) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone '{name}'")
    return backbone


class SelfAttention(nn.Module):
    """Single-head self-attention over the spatial feature map.

    Input:  (B, C, H, W) feature map from a CNN backbone.
    Output: (B, C, H, W) attention-refined feature map (residual connection).

    This lets distant spatial locations (e.g. two separate microaneurysms)
    influence each other's representation, which plain convolutions with
    limited receptive fields cannot do directly.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        n = height * width

        q = self.query(x).view(batch, -1, n).permute(0, 2, 1)  # (B, N, C//8)
        k = self.key(x).view(batch, -1, n)  # (B, C//8, N)
        v = self.value(x).view(batch, -1, n)  # (B, C, N)

        attention = self.softmax(torch.bmm(q, k))  # (B, N, N)
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch, channels, height, width)

        return self.gamma * out + x  # residual connection


class DRModel(nn.Module):
    """Single-image DR classifier: ResNet backbone (+ optional self-attention head)."""

    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        net = _load_backbone(backbone, pretrained)
        feature_dim = net.fc.in_features

        # Keep everything up to (not including) the global avg pool + fc,
        # so we still have a spatial feature map to run attention over.
        self.features = nn.Sequential(*list(net.children())[:-2])
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(feature_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        if self.use_attention:
            feat_map = self.attention(feat_map)
        pooled = self.pool(feat_map).flatten(1)
        return self.classifier(pooled)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Exposes the last conv feature map, used by Grad-CAM."""
        feat_map = self.features(x)
        if self.use_attention:
            feat_map = self.attention(feat_map)
        return feat_map


class DualInputDRModel(nn.Module):
    """Two-image variant: shares a backbone across both inputs, then fuses features."""

    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "resnet18",
        pretrained: bool = True,
        use_attention: bool = False,
    ):
        super().__init__()
        net = _load_backbone(backbone, pretrained)
        feature_dim = net.fc.in_features

        self.shared_features = nn.Sequential(*list(net.children())[:-2])
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(feature_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim * 2, num_classes)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.shared_features(x)
        if self.use_attention:
            feat_map = self.attention(feat_map)
        return self.pool(feat_map).flatten(1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self._encode(x1)
        f2 = self._encode(x2)
        fused = torch.cat([f1, f2], dim=1)
        return self.classifier(fused)


def build_model(config: dict) -> nn.Module:
    """Factory that builds the right model variant from a config dict."""
    model_cfg = config["model"]
    if config["data"].get("dual_image", False):
        return DualInputDRModel(
            num_classes=config["training"]["num_classes"],
            backbone=model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            use_attention=model_cfg["use_attention"],
        )
    return DRModel(
        num_classes=config["training"]["num_classes"],
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
        use_attention=model_cfg["use_attention"],
    )
