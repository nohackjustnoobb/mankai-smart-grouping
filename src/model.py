"""Siamese model components and deployment export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _FeatureBackbone(nn.Module):
    """Typing-only view of timm feature-extraction backbones."""

    num_features: int
    head_hidden_size: int


class ImageEncoder(nn.Module):
    """Encode one normalized 224x224 RGB image into a unit embedding."""

    def __init__(
        self,
        model_name: str = "mobilenetv3_large_100",
        embedding_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.backbone = cast(
            _FeatureBackbone,
            timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            ),
        )
        # MobileNetV3 reports its 1280-channel head separately from num_features.
        try:
            feature_dim = self.backbone.head_hidden_size
        except AttributeError:
            feature_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        features = self.backbone(image)
        embedding = self.projection(features)
        return F.normalize(embedding, p=2.0, dim=1)

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Freeze MobileNet while leaving the learned projection trainable."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen


class PairClassifier(nn.Module):
    """Order-invariant classifier operating on two image embeddings."""

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embedding_1: Tensor, embedding_2: Tensor) -> Tensor:
        # Symmetric features make the classifier order invariant.
        pair_features = torch.cat(
            (torch.abs(embedding_1 - embedding_2), embedding_1 * embedding_2),
            dim=1,
        )
        return self.network(pair_features).squeeze(1)


class SiameseNetwork(nn.Module):
    """Training-only composition of the shared encoder and pair classifier."""

    def __init__(self, encoder: ImageEncoder, classifier: PairClassifier) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, image_1: Tensor, image_2: Tensor) -> Tensor:
        batch_size = image_1.shape[0]
        embeddings = self.encoder(torch.cat((image_1, image_2), dim=0))
        embedding_1, embedding_2 = embeddings.split(batch_size, dim=0)
        return self.classifier(embedding_1, embedding_2)


def build_model(
    *,
    model_name: str = "mobilenetv3_large_100",
    embedding_dim: int = 256,
    classifier_hidden_dim: int = 256,
    dropout: float = 0.2,
    pretrained: bool = True,
) -> SiameseNetwork:
    encoder = ImageEncoder(model_name, embedding_dim, pretrained)
    classifier = PairClassifier(embedding_dim, classifier_hidden_dim, dropout)
    return SiameseNetwork(encoder, classifier)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], *, pretrained: bool = False
) -> SiameseNetwork:
    if checkpoint.get("stage") not in {"training", "finalized"}:
        raise ValueError("checkpoint stage must be training or finalized")
    config = checkpoint["config"]
    model = build_model(
        model_name=config["model_name"],
        embedding_dim=config["embedding_dim"],
        classifier_hidden_dim=config["classifier_hidden_dim"],
        dropout=config["dropout"],
        pretrained=pretrained,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def export_torchscript_models(
    model: SiameseNetwork,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write an encoder and a pair classifier that returns raw logits."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()

    encoder_path = output_dir / "image_encoder.pt"
    classifier_path = output_dir / "pair_classifier.pt"

    # Export both models without embedding deployment policy.
    example_image = torch.zeros(1, 3, 224, 224)
    scripted_encoder = torch.jit.trace(model.encoder, example_image)
    scripted_classifier = torch.jit.script(model.classifier)

    torch.jit.save(scripted_encoder, encoder_path)
    torch.jit.save(scripted_classifier, classifier_path)
    return encoder_path, classifier_path
