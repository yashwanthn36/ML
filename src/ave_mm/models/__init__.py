"""AVE convenience adapter over the dataset-agnostic model factory."""
from multimodal.models import TwoModalityFusion, build_model as build_generic_model


def build_model(num_classes: int, dim: int = 128, heads: int = 4,
                encoder_1: dict | None = None, encoder_2: dict | None = None,
                freeze_encoders: bool = False,
                cached_features: bool = False) -> TwoModalityFusion:
    """Build AVE's configured audio/video encoders through the generic factory."""
    return build_generic_model(
        num_classes, dim=dim, heads=heads,
        encoder_1=encoder_1 or {"name": "audio_cnn"},
        encoder_2=encoder_2 or {"name": "frame_cnn"},
        freeze_encoders=freeze_encoders,
        cached_features=cached_features,
    )


__all__ = ["build_model"]
