from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from moliv_rl.losses import FocalLoss


class TestFocalLoss:
    """Behavioral tests for FocalLoss."""

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
    def test_2d_shape_contract(self, reduction: str, gamma: float) -> None:
        criterion = FocalLoss(gamma=gamma, reduction=reduction)  # type: ignore[arg-type]
        inputs = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,), dtype=torch.int64)

        loss = criterion(inputs, targets)

        if reduction == "none":
            assert loss.shape == (4,)
        else:
            assert loss.shape == ()

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_dense_multidim_shape_contract(self, reduction: str) -> None:
        criterion = FocalLoss(gamma=2.0, reduction=reduction)  # type: ignore[arg-type]
        inputs = torch.randn(2, 4, 8, 8)
        targets = torch.randint(0, 4, (2, 8, 8), dtype=torch.int64)

        loss = criterion(inputs, targets)

        if reduction == "none":
            assert loss.shape == (2, 8, 8)
        else:
            assert loss.shape == ()

    def test_gamma_zero_matches_cross_entropy(self) -> None:
        criterion = FocalLoss(gamma=0.0, reduction="mean")
        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,), dtype=torch.int64)

        focal_val = criterion(inputs, targets)
        ce_val = F.cross_entropy(inputs, targets, reduction="mean")

        assert torch.allclose(focal_val, ce_val, atol=1e-5)

    def test_focal_weighting_downweights_easy_examples(self) -> None:
        criterion_ce = FocalLoss(gamma=0.0, reduction="none")
        criterion_focal = FocalLoss(gamma=2.0, reduction="none")

        inputs = torch.tensor(
            [
                [3.0, -3.0],
                [0.1, -0.1],
            ]
        )
        targets = torch.tensor([0, 0], dtype=torch.int64)

        loss_ce = criterion_ce(inputs, targets)
        loss_focal = criterion_focal(inputs, targets)

        # For the well-classified example (high pt), focal loss downweights more aggressively
        easy_ratio = (loss_focal[0] / loss_ce[0]).item()
        hard_ratio = (loss_focal[1] / loss_ce[1]).item()
        assert easy_ratio < hard_ratio

    def test_scalar_alpha_weighting(self) -> None:
        criterion_unscaled = FocalLoss(alpha=None, gamma=2.0, reduction="mean")
        criterion_scaled = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")

        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.int64)

        unscaled_loss = criterion_unscaled(inputs, targets)
        scaled_loss = criterion_scaled(inputs, targets)

        assert torch.allclose(scaled_loss, unscaled_loss * 0.5, atol=1e-6)

    def test_tensor_alpha_class_weights(self) -> None:
        weights = torch.tensor([1.0, 2.0, 0.5])
        criterion = FocalLoss(alpha=weights, gamma=0.0, reduction="none")

        inputs = torch.randn(6, 3)
        targets = torch.randint(0, 3, (6,), dtype=torch.int64)

        focal_val = criterion(inputs, targets)
        ce_val = F.cross_entropy(inputs, targets, weight=weights, reduction="none")

        assert torch.allclose(focal_val, ce_val, atol=1e-5)

    def test_ignore_index_behavior(self) -> None:
        criterion = FocalLoss(ignore_index=-100, reduction="none")
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, -100, 2, -100], dtype=torch.int64)

        loss = criterion(inputs, targets)
        assert loss[1].item() == 0.0
        assert loss[3].item() == 0.0
        assert loss[0].item() > 0.0
        assert loss[2].item() > 0.0

    def test_all_ignored_returns_zero(self) -> None:
        criterion = FocalLoss(ignore_index=-100, reduction="mean")
        inputs = torch.randn(3, 4)
        targets = torch.full((3,), -100, dtype=torch.int64)

        loss = criterion(inputs, targets)
        assert loss.item() == 0.0

    def test_label_smoothing_integration(self) -> None:
        criterion = FocalLoss(gamma=0.0, label_smoothing=0.1, reduction="mean")
        inputs = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,), dtype=torch.int64)

        focal_val = criterion(inputs, targets)
        ce_val = F.cross_entropy(
            inputs, targets, label_smoothing=0.1, reduction="mean"
        )
        assert torch.allclose(focal_val, ce_val, atol=1e-5)

    def test_gradient_flow(self) -> None:
        criterion = FocalLoss(gamma=2.0, reduction="mean")
        inputs = torch.randn(4, 5, requires_grad=True)
        targets = torch.randint(0, 5, (4,), dtype=torch.int64)

        loss = criterion(inputs, targets)
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()
        assert not torch.isinf(inputs.grad).any()

    # Parameter validation fail-fast tests
    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            FocalLoss(gamma=-0.5)

    def test_invalid_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="reduction must be one of"):
            FocalLoss(reduction="invalid")  # type: ignore[arg-type]

    def test_invalid_label_smoothing_raises(self) -> None:
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            FocalLoss(label_smoothing=1.5)
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            FocalLoss(label_smoothing=-0.1)

    def test_invalid_alpha_scalar_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            FocalLoss(alpha=-1.0)

    def test_invalid_alpha_type_raises(self) -> None:
        with pytest.raises(TypeError, match="alpha must be None"):
            FocalLoss(alpha="not_a_number")  # type: ignore[arg-type]

    def test_invalid_alpha_tensor_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="tensor alpha must have shape"):
            FocalLoss(alpha=torch.randn(3, 3))

    def test_negative_alpha_tensor_value_raises(self) -> None:
        with pytest.raises(ValueError, match="all tensor alpha values must be non-negative"):
            FocalLoss(alpha=torch.tensor([1.0, -0.5, 2.0]))

    # Input validation tests during forward
    def test_non_tensor_inputs_raise(self) -> None:
        criterion = FocalLoss()
        with pytest.raises(TypeError, match="inputs must be a torch.Tensor"):
            criterion([1, 2, 3], torch.tensor([0]))  # type: ignore[arg-type]

    def test_non_tensor_targets_raise(self) -> None:
        criterion = FocalLoss()
        with pytest.raises(TypeError, match="targets must be a torch.Tensor"):
            criterion(torch.randn(2, 3), [0, 1])  # type: ignore[arg-type]

    def test_inputs_less_than_2d_raises(self) -> None:
        criterion = FocalLoss()
        with pytest.raises(ValueError, match="inputs must have shape"):
            criterion(torch.randn(5), torch.tensor(0))

    def test_targets_non_integer_dtype_raises(self) -> None:
        criterion = FocalLoss()
        with pytest.raises(TypeError, match="targets must contain integer"):
            criterion(torch.randn(2, 3), torch.tensor([0.0, 1.0]))

    def test_batch_size_mismatch_raises(self) -> None:
        criterion = FocalLoss()
        with pytest.raises(ValueError, match="Batch size mismatch"):
            criterion(torch.randn(4, 3), torch.tensor([0, 1]))

    def test_target_shape_mismatch_dense_raises(self) -> None:
        criterion = FocalLoss()
        with pytest.raises(ValueError, match="targets must have shape"):
            criterion(torch.randn(2, 3, 4, 4), torch.randint(0, 3, (2, 4, 5)))

    def test_alpha_numel_mismatch_raises(self) -> None:
        criterion = FocalLoss(alpha=torch.tensor([1.0, 1.0, 1.0]))
        with pytest.raises(ValueError, match="alpha must contain one value per class"):
            criterion(torch.randn(2, 5), torch.tensor([0, 1]))
