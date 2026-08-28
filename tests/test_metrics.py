from __future__ import annotations

import pytest
import torch

from moliv_rl.metrics import (
    AverageMeter,
    calculate_accuracy,
    calculate_precision,
)


class TestAverageMeter:
    """Behavioral tests for AverageMeter."""

    def test_update_single_values(self) -> None:
        meter = AverageMeter()
        assert meter.avg == 0.0
        assert meter.count == 0

        meter.update(2.0)
        assert meter.val == 2.0
        assert meter.avg == 2.0
        assert meter.sum == 2.0
        assert meter.count == 1

        meter.update(4.0)
        assert meter.val == 4.0
        assert meter.avg == 3.0
        assert meter.sum == 6.0
        assert meter.count == 2

    def test_update_weighted(self) -> None:
        meter = AverageMeter()
        meter.update(10.0, n=2)
        meter.update(20.0, n=3)
        assert meter.val == 20.0
        assert meter.count == 5
        assert meter.sum == 80.0
        assert meter.avg == 16.0

    def test_update_zero_count_noop(self) -> None:
        meter = AverageMeter()
        meter.update(5.0, n=1)
        meter.update(100.0, n=0)
        assert meter.val == 5.0
        assert meter.count == 1
        assert meter.avg == 5.0

    def test_reset(self) -> None:
        meter = AverageMeter()
        meter.update(10.0, n=5)
        meter.reset()
        assert meter.val == 0.0
        assert meter.avg == 0.0
        assert meter.sum == 0.0
        assert meter.count == 0


class TestCalculateAccuracy:
    """Behavioral tests for calculate_accuracy."""

    def test_perfect_accuracy(self) -> None:
        outputs = torch.tensor(
            [
                [10.0, 1.0, 0.0],
                [0.0, 10.0, 1.0],
                [0.0, 1.0, 10.0],
            ]
        )
        targets = torch.tensor([0, 1, 2], dtype=torch.int64)
        acc = calculate_accuracy(outputs, targets)
        assert acc == 1.0

    def test_zero_accuracy(self) -> None:
        outputs = torch.tensor(
            [
                [10.0, 1.0, 0.0],
                [0.0, 10.0, 1.0],
            ]
        )
        targets = torch.tensor([1, 0], dtype=torch.int64)
        acc = calculate_accuracy(outputs, targets)
        assert acc == 0.0

    def test_fractional_accuracy(self) -> None:
        outputs = torch.tensor(
            [
                [5.0, 1.0],
                [1.0, 5.0],
                [5.0, 1.0],
                [1.0, 5.0],
            ]
        )
        targets = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
        acc = calculate_accuracy(outputs, targets)
        assert acc == 0.5

    def test_empty_batch_returns_zero(self) -> None:
        outputs = torch.empty((0, 5))
        targets = torch.empty((0,), dtype=torch.int64)
        assert calculate_accuracy(outputs, targets) == 0.0

    def test_outputs_wrong_ndim_raises(self) -> None:
        with pytest.raises((IndexError, RuntimeError)):
            calculate_accuracy(torch.randn(4), torch.tensor([0, 1, 2, 3]))

    def test_batch_size_mismatch_raises(self) -> None:
        with pytest.raises(RuntimeError):
            calculate_accuracy(torch.randn(4, 3), torch.tensor([0, 1]))


class TestCalculatePrecision:
    """Behavioral tests for calculate_precision."""

    def test_macro_precision(self) -> None:
        # Predictions: [0, 0, 1, 2]
        # Targets:     [0, 1, 1, 2]
        # Class 0: TP=1, FP=1 -> Prec=0.5
        # Class 1: TP=1, FP=0 -> Prec=1.0
        # Class 2: TP=1, FP=0 -> Prec=1.0
        # Macro: (0.5 + 1.0 + 1.0) / 3 = 0.8333...
        outputs = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        targets = torch.tensor([0, 1, 1, 2], dtype=torch.int64)

        macro_prec = calculate_precision(outputs, targets, average="macro")
        assert isinstance(macro_prec, float)
        assert pytest.approx(macro_prec, rel=1e-4) == (0.5 + 1.0 + 1.0) / 3.0

    def test_micro_precision(self) -> None:
        # Total TP = 3, Total Predicted Positives = 4
        # Micro = 3 / 4 = 0.75
        outputs = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        targets = torch.tensor([0, 1, 1, 2], dtype=torch.int64)

        micro_prec = calculate_precision(outputs, targets, average="micro")
        assert isinstance(micro_prec, float)
        assert pytest.approx(micro_prec, rel=1e-4) == 0.75

    def test_weighted_precision(self) -> None:
        # Supports: Class 0=1, Class 1=2, Class 2=1 (Total = 4)
        # Class 0 prec=0.5, Class 1 prec=1.0, Class 2 prec=1.0
        # Weighted = (0.5*1 + 1.0*2 + 1.0*1) / 4 = 3.5 / 4 = 0.875
        outputs = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        targets = torch.tensor([0, 1, 1, 2], dtype=torch.int64)

        weighted_prec = calculate_precision(outputs, targets, average="weighted")
        assert isinstance(weighted_prec, float)
        assert pytest.approx(weighted_prec, rel=1e-4) == 0.875

    def test_per_class_precision(self) -> None:
        outputs = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        targets = torch.tensor([0, 1, 1, 2], dtype=torch.int64)

        per_class = calculate_precision(outputs, targets, average=None)
        assert isinstance(per_class, torch.Tensor)
        assert per_class.shape == (3,)
        expected = torch.tensor([0.5, 1.0, 1.0], dtype=torch.float32)
        assert torch.allclose(per_class, expected, atol=1e-5)

    def test_zero_division_behavior(self) -> None:
        # Class 1 is never predicted
        outputs = torch.tensor(
            [
                [10.0, 0.0],
                [10.0, 0.0],
            ]
        )
        targets = torch.tensor([0, 1], dtype=torch.int64)

        prec_0 = calculate_precision(outputs, targets, average=None, zero_division=0.0)
        assert isinstance(prec_0, torch.Tensor)
        assert prec_0[1].item() == 0.0

        prec_1 = calculate_precision(outputs, targets, average=None, zero_division=1.0)
        assert isinstance(prec_1, torch.Tensor)
        assert prec_1[1].item() == 1.0

    def test_empty_targets_returns_zero_division(self) -> None:
        outputs = torch.empty((0, 3))
        targets = torch.empty((0,), dtype=torch.int64)

        res_scalar = calculate_precision(outputs, targets, average="macro", zero_division=0.0)
        assert res_scalar == 0.0

        res_tensor = calculate_precision(outputs, targets, average=None, zero_division=1.0)
        assert isinstance(res_tensor, torch.Tensor)
        assert res_tensor.shape == (3,)
        assert (res_tensor == 1.0).all()

    def test_dimension_and_range_mismatch_raises(self) -> None:
        with pytest.raises((IndexError, RuntimeError)):
            calculate_precision(torch.randn(2), torch.tensor([0]))

        with pytest.raises((RuntimeError, IndexError)):
            calculate_precision(torch.randn(2, 2), torch.tensor([0, 5]))
