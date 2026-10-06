import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetT2IAdapter(nn.Module):
    """
    Adaptateur T2I MobileNetV3-Small multi-échelle pour FLUX.1.
    Extrait les caractéristiques spatiales à plusieurs échelles (strides 4, 8, 16, 32),
    les fusionne via interpolation bilinéaire et les projette dans l'espace de tokens FLUX.
    """

    def __init__(
        self,
        flux_hidden_dim: int = 3072,
        num_in_channels: int = 3,
        adapter_dim: int = 768,
        pretrained: bool = True,
    ):
        super().__init__()
        self.flux_hidden_dim = flux_hidden_dim
        self.adapter_dim = adapter_dim

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_mobilenet = mobilenet_v3_small(weights=weights)

        if num_in_channels != 3:
            old_conv = base_mobilenet.features[0][0]
            base_mobilenet.features[0][0] = nn.Conv2d(
                num_in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        # Décomposition multi-échelle :
        # stage 1 : stride 4, 24 canaux (haute résolution spatiale pour les contours)
        # stage 2 : stride 8, 48 canaux
        # stage 3 : stride 16, 96 canaux
        # stage 4 : stride 32, 576 canaux (haute sémantique)
        self.stage1 = base_mobilenet.features[:4]
        self.stage2 = base_mobilenet.features[4:9]
        self.stage3 = base_mobilenet.features[9:12]
        self.stage4 = base_mobilenet.features[12:]

        # Projections latérales 1x1
        self.lat1 = nn.Conv2d(24, 128, kernel_size=1)
        self.lat2 = nn.Conv2d(48, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(96, 256, kernel_size=1)
        self.lat4 = nn.Conv2d(576, 256, kernel_size=1)

        # Réseau de fusion final
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 128 + 256 + 256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, adapter_dim, kernel_size=1),
        )

    @property
    def features(self):
        """Propriété de rétrocompatibilité pour accéder aux blocs sous forme de séquence."""
        return nn.Sequential(
            *self.stage1,
            *self.stage2,
            *self.stage3,
            *self.stage4,
        )

    @property
    def conv_out(self):
        """Propriété de rétrocompatibilité pointant vers la couche de projection finale."""
        return self.fusion[2]

    def forward(self, condition_image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Args:
            condition_image: [B, C, H, W] image de conditionnement
            target_h: hauteur spatiale cible pour les tokens de sortie (grille latente FLUX)
            target_w: largeur spatiale cible pour les tokens de sortie (grille latente FLUX)

        Returns:
            adapter_tokens: [B, target_h * target_w, adapter_dim]
        """
        x1 = self.stage1(condition_image)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        p1 = F.interpolate(self.lat1(x1), size=(target_h, target_w), mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.lat2(x2), size=(target_h, target_w), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.lat3(x3), size=(target_h, target_w), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.lat4(x4), size=(target_h, target_w), mode="bilinear", align_corners=False)

        feat = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))

        B, C, H, W = feat.shape
        adapter_tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return adapter_tokens
