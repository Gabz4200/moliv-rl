"""moliv-rl: Autonomous game-playing agent research package."""

from __future__ import annotations

from .data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DataLoaderConfig,
    StreamingGameQADataset,
    StreamingGameQADatasetConfig,
    get_dataloaders,
    get_default_datasets,
    get_streaming_gameqa_datasets,
    get_train_transforms,
    get_val_transforms,
)
from .losses import FocalLoss, Reduction
from .metrics import (
    AverageMeter,
    PrecisionAverage,
    calculate_accuracy,
    calculate_precision,
)
from .models import (
    MODEL_REGISTRY,
    ClassificationModel,
    GatedConv2D,
    HardSigmoid,
    HardSwish,
    InvertedResidual,
    LiVConv2D,
    MLPConv2D,
    MobileNetV3,
    MyBlock,
    MyModel,
    MyVideoModel,
    SEModule,
    get_model,
    mobilenetv3_large,
    mobilenetv3_small,
)
from .train import ClassificationTrainer
from .utils import get_logger, seed_worker, set_seeds

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MODEL_REGISTRY",
    "AverageMeter",
    "ClassificationModel",
    "ClassificationTrainer",
    "DataLoaderConfig",
    "FocalLoss",
    "GatedConv2D",
    "HardSigmoid",
    "HardSwish",
    "InvertedResidual",
    "LiVConv2D",
    "MLPConv2D",
    "MobileNetV3",
    "MyBlock",
    "MyModel",
    "MyVideoModel",
    "PrecisionAverage",
    "Reduction",
    "SEModule",
    "StreamingGameQADataset",
    "StreamingGameQADatasetConfig",
    "calculate_accuracy",
    "calculate_precision",
    "get_dataloaders",
    "get_default_datasets",
    "get_logger",
    "get_model",
    "get_streaming_gameqa_datasets",
    "get_train_transforms",
    "get_val_transforms",
    "mobilenetv3_large",
    "mobilenetv3_small",
    "seed_worker",
    "set_seeds",
]
