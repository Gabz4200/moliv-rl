from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms


@dataclass
class DataLoaderConfig:
    """Configuration container for :func:`get_dataloaders`.

    Grouping DataLoader options here keeps the public factory signature stable
    as new options are added.
    """

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool | None = None
    generator: torch.Generator | None = None
    worker_init_fn: Any = None
    data_dir: Path | str = "data"
    image_size: int = 64


def get_default_datasets(
    data_dir: Path | str = "data",
    image_size: int = 64,
) -> tuple[ImageFolder, ImageFolder]:
    r"""get_default_datasets(data_dir='data', image_size=64) -> Tuple[ImageFolder, ImageFolder]

    Create train and validation :class:`~torchvision.datasets.ImageFolder` datasets on demand.

    Args:
        data_dir (Path or str, optional): Root directory path containing ``'train'`` and ``'val'`` subdirectories. Default: ``'data'``
        image_size (int, optional): Spatial image resolution for dataset transform pipelines. Default: ``64``

    Returns:
        tuple: A pair ``(train_dataset, val_dataset)`` of instantiated :class:`~torchvision.datasets.ImageFolder` objects.
    """
    root = Path(data_dir)
    train_ds = ImageFolder(
        root=str(root / "train"),
        transform=get_train_transforms(image_size=image_size),
    )
    val_ds = ImageFolder(
        root=str(root / "val"),
        transform=get_val_transforms(image_size=image_size),
    )
    return train_ds, val_ds


def get_dataloaders(
    config: DataLoaderConfig,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
) -> tuple[DataLoader, DataLoader]:
    r"""get_dataloaders(config, train_dataset=None, val_dataset=None) -> Tuple[DataLoader, DataLoader]

    Create hardware-optimized :class:`~torch.utils.data.DataLoader` instances for training and validation.

    Args:
        config (DataLoaderConfig): Container for DataLoader hyperparameters and defaults.
        train_dataset (Dataset, optional): Custom training dataset. If ``None``, loads from :attr:`~DataLoaderConfig.data_dir`. Default: ``None``
        val_dataset (Dataset, optional): Custom validation dataset. If ``None``, loads from :attr:`~DataLoaderConfig.data_dir`. Default: ``None``

    Returns:
        tuple: A pair ``(train_loader, val_loader)`` of configured :class:`~torch.utils.data.DataLoader` objects.
    """
    train_ds = train_dataset
    val_ds = val_dataset

    if train_ds is None or val_ds is None:
        default_train_ds, default_val_ds = get_default_datasets(
            data_dir=config.data_dir,
            image_size=config.image_size,
        )
        if train_ds is None:
            train_ds = default_train_ds
        if val_ds is None:
            val_ds = default_val_ds

    if config.num_workers > 0:
        use_persistent = (
            config.persistent_workers
            if config.persistent_workers is not None
            else True
        )
    else:
        use_persistent = False

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=not isinstance(train_ds, IterableDataset),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=use_persistent,
        generator=config.generator,
        worker_init_fn=config.worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=use_persistent,
        worker_init_fn=config.worker_init_fn,
    )

    return train_loader, val_loader


__all__ = [
    "StreamingGameQADataset",
    "StreamingGameQADatasetConfig",
    "get_dataloaders",
    "get_default_datasets",
    "get_streaming_gameqa_datasets",
]


@dataclass
class StreamingGameQADatasetConfig:
    """Configuration for loading the GameQA-140K dataset with streaming.

    The defaults are tuned for ``OpenMOSS-Team/GameQA-140K`` but the
    configuration is generic enough to load any Hugging Face dataset that
    exposes an image column and a categorical label column.
    """

    dataset_id: str = "OpenMOSS-Team/GameQA-140K"
    split: str = "train"
    image_column: str = "image"
    label_column: str = "game_name"
    image_size: int = 256
    train: bool = True
    label2id: dict[str, int] | None = None
    cache_dir: Path | str | None = None
    token: bool | str | None = None
    streaming: bool = True
    max_samples: int | None = None
    shuffle_buffer_size: int = 10_000
    shuffle_seed: int = 42
    trust_remote_code: bool = False
    transform: Callable | None = field(default=None, repr=False)


class StreamingGameQADataset(IterableDataset):
    """Streaming dataset wrapper for Hugging Face game-QA datasets.

    Loads samples from a remote Hugging Face dataset without materializing
    the full split in memory.  Only the ``image`` and ``game_name`` columns
    are retained by default.
    """

    def __init__(self, config: StreamingGameQADatasetConfig) -> None:
        super().__init__()
        self.config = config
        self._transform = config.transform
        self._image_column = config.image_column
        self._label_column = config.label_column
        self._image_size = config.image_size

        if self._transform is None:
            if self.config.train:
                self._transform = get_train_transforms(image_size=self.config.image_size)
            else:
                self._transform = get_val_transforms(image_size=self.config.image_size)

        self._label2id: dict[str, int] = dict(config.label2id) if config.label2id else {}
        self._id2label: dict[int, str] = {v: k for k, v in self._label2id.items()}
        self._next_label_id = len(self._label2id)
        self._load_dataset()

    def _load_dataset(self) -> None:
        from datasets import load_dataset

        self._hf_dataset = load_dataset(
            self.config.dataset_id,
            split=self.config.split,
            streaming=self.config.streaming,
            cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
            token=self.config.token,
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.config.train and self.config.streaming:
            self._hf_dataset = self._hf_dataset.shuffle(
                buffer_size=self.config.shuffle_buffer_size,
                seed=self.config.shuffle_seed,
            )

        if self.config.max_samples is not None:
            self._hf_dataset = self._hf_dataset.take(self.config.max_samples)

        self._hf_dataset = self._hf_dataset.select_columns(
            [self.config.image_column, self.config.label_column]
        )

        features = getattr(self._hf_dataset, "features", None)
        if features and self.config.label_column in features:
            feature = features[self.config.label_column]
            if hasattr(feature, "names"):
                for idx, name in enumerate(feature.names):
                    self._label2id.setdefault(name, idx)
                if self._label2id:
                    self._next_label_id = max(self._label2id.values()) + 1
                    self._id2label = {v: k for k, v in self._label2id.items()}

    def __iter__(self) -> Any:
        worker_info = torch.utils.data.get_worker_info()
        dataset = self._hf_dataset

        if (
            worker_info is not None
            and worker_info.num_workers > 1
            and hasattr(dataset, "shard")
        ):
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        for example in dataset:
            example = cast(dict[str, Any], example)
            image = example[self._image_column]
            label = example[self._label_column]

            if isinstance(image, dict):
                if "bytes" in image:
                    image = Image.open(io.BytesIO(image["bytes"]))
                elif "path" in image:
                    image = Image.open(image["path"]).convert("RGB")
                else:
                    raise ValueError(f"Unexpected image dict keys: {list(image.keys())}")
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")
            else:
                image = image.convert("RGB")

            if self._transform is not None:
                image = self._transform(image)

            if label not in self._label2id:
                self._label2id[label] = self._next_label_id
                self._id2label[self._next_label_id] = label
                self._next_label_id += 1

            label_id = self._label2id[label]
            yield (image, torch.tensor(label_id, dtype=torch.long))

    def get_label_name(self, label_id: int) -> str:
        """Return the human-readable game name for a numeric class index."""
        return self._id2label.get(label_id, "<unknown>")

    def __len__(self) -> int:
        if self.config.max_samples is not None:
            return self.config.max_samples
        raise TypeError(
            "StreamingGameQADataset does not have a known length when max_samples is None."
        )

    @property
    def label2id(self) -> dict[str, int]:
        """Mapping from ``game_name`` strings to class indices."""
        return dict(self._label2id)

    @property
    def id2label(self) -> dict[int, str]:
        """Mapping from class indices back to ``game_name`` strings."""
        return dict(self._id2label)


def get_streaming_gameqa_datasets(
    image_size: int = 256,
    train_split: str = "preview",
    val_split: str | None = "preview",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[StreamingGameQADataset, StreamingGameQADataset | None]:
    """Create streaming train/val datasets for GameQA-140K.

    Args:
        image_size: Target spatial resolution for the image pipeline.
        train_split: Hugging Face split name for training data.
        val_split: Hugging Face split name for validation data.  Pass
            ``None`` to skip the validation dataset.
        max_train_samples: Optional limit on the number of training samples.
        max_val_samples: Optional limit on the number of validation samples.

    Returns:
        A ``(train_dataset, val_dataset)`` pair.  ``val_dataset`` is
        ``None`` when ``val_split`` is ``None``.
    """
    train_config = StreamingGameQADatasetConfig(
        split=train_split,
        image_size=image_size,
        train=True,
        max_samples=max_train_samples,
    )
    train_dataset = StreamingGameQADataset(config=train_config)

    val_dataset = None
    if val_split is not None:
        val_config = StreamingGameQADatasetConfig(
            split=val_split,
            image_size=image_size,
            train=False,
            label2id=train_dataset.label2id,
            max_samples=max_val_samples,
        )
        val_dataset = StreamingGameQADataset(config=val_config)

    return train_dataset, val_dataset
