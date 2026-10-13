from .fusion import TwoModalityFusion
from .encoders import (AudioCNNEncoder, FrameCNNEncoder, PANNsCnn14AudioEncoder,
                       ResNet18FrameEncoder, build_encoder)


def build_model(num_classes: int, *, encoder_1: dict | None = None, encoder_2: dict | None = None,
                dim: int = 128, heads: int = 4,
                freeze_encoders: bool = False,
                cached_features: bool = False) -> TwoModalityFusion:
    """Compose registered encoders with the dataset-agnostic fusion head."""
    if cached_features:
        from torch import nn
        first, second = nn.Identity(), nn.Identity()
        freeze_encoders = False
    else:
        encoder_1, encoder_2 = encoder_1 or {}, encoder_2 or {}
        first = build_encoder(encoder_1.get("name", "custom"), dim=dim,
                              kwargs=encoder_1.get("kwargs"), factory=encoder_1.get("factory"))
        second = build_encoder(encoder_2.get("name", "custom"), dim=dim,
                               kwargs=encoder_2.get("kwargs"), factory=encoder_2.get("factory"))
    return TwoModalityFusion(first, second, num_classes, dim=dim, heads=heads,
                             freeze_encoders=freeze_encoders)


__all__ = ["TwoModalityFusion", "AudioCNNEncoder", "PANNsCnn14AudioEncoder", "FrameCNNEncoder",
           "ResNet18FrameEncoder", "build_encoder", "build_model"]
