import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Small_Weights
from config import PART_CLASSES, DAMAGE_CLASSES

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, feature_dim * 2),
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.constant_(self.modulator[-1].bias[:feature_dim], 1.0)
        nn.init.constant_(self.modulator[-1].bias[feature_dim:], 0.0)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        params = self.modulator(condition)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta
    
class ConvNeXtWithFiLM(nn.Module):
    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(self, embedding_dim: int = 64, pretrained: bool = False):
        super().__init__()
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.convnext_small(weights=weights)

        self.stem = base.features[0]
        self.stage0 = base.features[1]
        self.down1 = base.features[2]
        self.stage1 = base.features[3]
        self.down2 = base.features[4]
        self.stage2 = base.features[5]
        self.down3 = base.features[6]
        self.stage3 = base.features[7]
        self.avgpool = base.avgpool

        self.film1 = FiLMLayer(embedding_dim, self.STAGE_CHANNELS[1])
        self.film2 = FiLMLayer(embedding_dim, self.STAGE_CHANNELS[2])
        self.film3 = FiLMLayer(embedding_dim, self.STAGE_CHANNELS[3])
        self.out_features = self.STAGE_CHANNELS[3]

    def forward(self, image: torch.Tensor, part_embedding: torch.Tensor) -> torch.Tensor:
        features = self.stem(image)
        features = self.stage0(features)

        features = self.down1(features)
        features = self.stage1(features)
        features = self.film1(features, part_embedding)

        features = self.down2(features)
        features = self.stage2(features)
        features = self.film2(features, part_embedding)

        features = self.down3(features)
        features = self.stage3(features)
        features = self.film3(features, part_embedding)

        features = self.avgpool(features)
        return features.flatten(1)
    
class DamageClassifierFiLM(nn.Module):
    def __init__(
        self,
        num_parts: int = len(PART_CLASSES),
        num_damage: int = len(DAMAGE_CLASSES),
        embedding_dim: int = 64,
        late_fusion: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()

        self.late_fusion = late_fusion
        self.part_embedding = nn.Embedding(num_parts, embedding_dim)
        nn.init.normal_(self.part_embedding.weight, std=0.01)

        self.backbone = ConvNeXtWithFiLM(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
        )

        classifier_in = self.backbone.out_features
        if late_fusion:
            classifier_in += embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_damage),
        )

    def forward(self, image: torch.Tensor, part_id: torch.Tensor) -> torch.Tensor:
        part_embedding = self.part_embedding(part_id)
        features = self.backbone(image, part_embedding)
        if self.late_fusion:
            features = torch.cat([features, part_embedding], dim=1)
        return self.classifier(features)