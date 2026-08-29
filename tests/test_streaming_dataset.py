from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from moliv_rl.data import (
    StreamingGameQADataset,
    StreamingGameQADatasetConfig,
    get_streaming_gameqa_datasets,
)


def _make_mock_example(label: str = "test_game") -> dict:
    return {
        "image": Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)),
        "game_name": label,
    }


def _mock_streaming_dataset(*args, **kwargs):
    examples = [_make_mock_example(f"game_{i}") for i in range(4)]

    class MockDataset:
        def select_columns(self, *args, **kwargs):
            return self

        def shuffle(self, *args, **kwargs):
            return self

        def take(self, n: int):
            self._take_n = n
            return self

        def shard(self, *args, **kwargs):
            return self

        @property
        def features(self):
            features = MagicMock()
            feature = MagicMock()
            feature.names = ["game_0", "game_1", "game_2", "game_3"]
            features.__getitem__ = lambda self, key: feature if key == "game_name" else None
            return features

        def __iter__(self):
            n = getattr(self, "_take_n", None)
            if n is not None:
                return iter(examples[:n])
            return iter(examples)

    return MockDataset()


class TestStreamingGameQADatasetConfig:
    def test_defaults(self) -> None:
        config = StreamingGameQADatasetConfig()
        assert config.dataset_id == "OpenMOSS-Team/GameQA-140K"
        assert config.split == "train"
        assert config.image_column == "image"
        assert config.label_column == "game_name"
        assert config.image_size == 256
        assert config.train is True
        assert config.streaming is True
        assert config.max_samples is None
        assert config.label2id is None

    def test_custom_values(self) -> None:
        config = StreamingGameQADatasetConfig(
            dataset_id="custom/dataset",
            split="test",
            image_size=128,
            train=False,
            max_samples=100,
            label2id={"a": 0, "b": 1},
        )
        assert config.dataset_id == "custom/dataset"
        assert config.split == "test"
        assert config.image_size == 128
        assert config.train is False
        assert config.max_samples == 100
        assert config.label2id == {"a": 0, "b": 1}


class TestStreamingGameQADataset:
    def test_len_with_max_samples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "datasets.load_dataset",
            lambda *args, **kwargs: _mock_streaming_dataset(),
        )
        config = StreamingGameQADatasetConfig(
            split="preview",
            image_size=32,
            train=True,
            max_samples=16,
        )
        dataset = StreamingGameQADataset(config=config)
        assert len(dataset) == 16

    def test_len_without_max_samples_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "datasets.load_dataset",
            lambda *args, **kwargs: _mock_streaming_dataset(),
        )
        config = StreamingGameQADatasetConfig(
            split="preview",
            image_size=32,
            train=True,
            max_samples=None,
        )
        dataset = StreamingGameQADataset(config=config)
        with pytest.raises(TypeError, match="max_samples is None"):
            _ = len(dataset)

    def test_iteration_returns_tensors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "datasets.load_dataset",
            lambda *args, **kwargs: _mock_streaming_dataset(),
        )

        config = StreamingGameQADatasetConfig(
            split="preview",
            image_size=32,
            train=True,
            max_samples=2,
        )
        dataset = StreamingGameQADataset(config=config)
        samples = list(dataset)
        assert len(samples) == 2
        image, label = samples[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long

    def test_label_mapping_properties(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "datasets.load_dataset",
            lambda *args, **kwargs: _mock_streaming_dataset(),
        )

        config = StreamingGameQADatasetConfig(
            split="preview",
            image_size=32,
            train=True,
            max_samples=2,
        )
        dataset = StreamingGameQADataset(config=config)
        list(dataset)  # consume to populate labels
        assert len(dataset.label2id) > 0
        assert len(dataset.id2label) > 0
        assert set(dataset.label2id.keys()) == set(dataset.id2label.values())
        assert set(dataset.id2label.keys()) == set(dataset.label2id.values())

    def test_get_label_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "datasets.load_dataset",
            lambda *args, **kwargs: _mock_streaming_dataset(),
        )

        config = StreamingGameQADatasetConfig(
            split="preview",
            image_size=32,
            train=True,
            max_samples=2,
        )
        dataset = StreamingGameQADataset(config=config)
        list(dataset)  # consume to populate labels
        if dataset.label2id:
            some_label = next(iter(dataset.label2id))
            label_id = dataset.label2id[some_label]
            assert dataset.get_label_name(label_id) == some_label
            assert dataset.get_label_name(9999) == "<unknown>"

    def test_get_streaming_gameqa_datasets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "datasets.load_dataset",
            lambda *args, **kwargs: _mock_streaming_dataset(),
        )

        train_ds, val_ds = get_streaming_gameqa_datasets(
            image_size=32,
            train_split="preview",
            val_split="preview",
            max_train_samples=2,
            max_val_samples=2,
        )
        assert isinstance(train_ds, StreamingGameQADataset)
        assert isinstance(val_ds, StreamingGameQADataset)
        assert len(train_ds) == 2
        assert len(val_ds) == 2
        assert train_ds.label2id == val_ds.label2id
