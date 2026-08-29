from __future__ import annotations

from typing import Literal

import torch
from torch import nn

LossName = Literal["cross_entropy", "focal_loss", "focal", "invariance"]


class SIGReg(nn.Module):
    r"""SIGReg(sketch_dim=64, num_integration_points=17, integration_t_max=3.0)

    Strong Sketched Isotropic Gaussian Regularization from LeJEPA.

    Constrains projected embeddings to match the empirical characteristic
    function (ECF) of an isotropic Gaussian using random projection and
    numerical integration. The projection directions are regenerated every
    forward call to approximate a full multivariate test.

    Args:
        sketch_dim (int, optional): Number of random 1D projections (slices).
            Default: ``64``
        num_integration_points (int, optional): Number of quadrature points for
            numerical integration of the ECF distance. Default: ``17``
        integration_t_max (float, optional): Upper bound for the integration
            domain :math:`[0, t_{\max}]`. Default: ``3.0``

    Shape:
        - Input: :math:`(N, D)` where :math:`N` is the number of samples and
          :math:`D` is the embedding dimension.
        - Output: Scalar loss value.

    Examples::

        >>> sigreg = SIGReg()
        >>> embeddings = torch.randn(256, 128, requires_grad=True)
        >>> loss = sigreg(embeddings)
        >>> loss.backward()
    """

    t: torch.Tensor
    phi: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self,
        sketch_dim: int = 64,
        num_integration_points: int = 17,
        integration_t_max: float = 3.0,
    ) -> None:
        super().__init__()

        self.sketch_dim = sketch_dim
        self.num_integration_points = num_integration_points
        self.integration_t_max = integration_t_max

        t = torch.linspace(
            0,
            float(integration_t_max),
            num_integration_points,
            dtype=torch.float32,
        )
        dt = float(integration_t_max) / (num_integration_points - 1)
        weights = torch.full((num_integration_points,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-(t**2) / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        r"""Compute the SIGReg loss for a batch of projected embeddings.

        Args:
            proj (torch.Tensor): Embeddings tensor of shape :math:`(N, D)`.

        Returns:
            torch.Tensor: Scalar SIGReg loss.
        """
        if proj.dim() != 2:
            raise ValueError(
                f"SIGReg expects a 2D tensor of shape (N, D), got shape {proj.shape}."
            )

        N, D = proj.size()

        if D > self.sketch_dim:
            A = torch.randn(
                D,
                self.sketch_dim,
                device=proj.device,
                dtype=proj.dtype,
            )
            A = A.div_(A.norm(p=2, dim=0, keepdim=True))
            x_t = (proj @ A).unsqueeze(-1) * self.t.to(proj.device)
        else:
            t = self.t.to(proj.device)
            x_t = proj.unsqueeze(-1) * t

        # Empirical characteristic function via cosine and sine moments
        ecf_cos = x_t.cos().mean(dim=-3)
        ecf_sin = x_t.sin().mean(dim=-3)

        # Distance to standard Gaussian ECF
        phi = self.phi.to(proj.device)
        weights = self.weights.to(proj.device)

        err = (ecf_cos - phi).square() + ecf_sin.square()
        statistic = (err @ weights) * float(N)

        return statistic.mean()


class WeakSIGReg(nn.Module):
    r"""WeakSIGReg(sketch_dim=64)

    Weak Sketched Isotropic Gaussian Regularization.

    Forces the empirical covariance of the embeddings to be close to the
    identity matrix (spherical cloud). Cheaper than :class:`SIGReg` while
    retaining most of the stabilization effect.

    Args:
        sketch_dim (int, optional): Maximum sketch dimension for random
            projection. If the embedding dimension is smaller than this value,
            the original dimension is used. Default: ``64``

    Shape:
        - Input: :math:`(N, D)` where :math:`N` is the number of samples and
          :math:`D` is the embedding dimension.
        - Output: Scalar loss value.

    Examples::

        >>> weak_sigreg = WeakSIGReg()
        >>> embeddings = torch.randn(256, 128, requires_grad=True)
        >>> loss = weak_sigreg(embeddings)
        >>> loss.backward()
    """

    def __init__(self, sketch_dim: int = 64) -> None:
        super().__init__()
        self.sketch_dim = sketch_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the WeakSIGReg loss for a batch of embeddings.

        Args:
            x (torch.Tensor): Embeddings tensor of shape :math:`(N, D)`.

        Returns:
            torch.Tensor: Scalar WeakSIGReg loss.
        """
        if x.dim() != 2:
            raise ValueError(
                f"WeakSIGReg expects a 2D tensor of shape (N, D), got shape {x.shape}."
            )

        N, C = x.size()

        if C > self.sketch_dim:
            S = torch.randn(
                self.sketch_dim,
                C,
                device=x.device,
                dtype=x.dtype,
            ) / (C**0.5)
            x = x @ S.T

        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (N - 1 + 1e-6)
        target = torch.eye(x.size(1), device=x.device, dtype=x.dtype)

        return torch.norm(cov - target, p="fro")


class LeJepaLoss(nn.Module):
    r"""LeJepaLoss(sigreg_loss_fn, lamb=0.02, normalize_projections=True)

    LeJEPA loss combining an invariance/prediction term with SIGReg.

    The LeJEPA objective is:

    .. math::

        \mathcal{L}_{\text{LeJEPA}} =
        (1 - \lambda) \mathcal{L}_{\text{inv}} +
        \lambda \mathcal{L}_{\text{SIGReg}}

    where :math:`\mathcal{L}_{\text{inv}}` is the mean-squared difference
    between each projected view and the batch-mean projection, and
    :math:`\mathcal{L}_{\text{SIGReg}}` constrains the projected embeddings
    to follow an isotropic Gaussian.

    Args:
        sigreg_loss_fn (nn.Module): SIGReg loss module, e.g.
            :class:`SIGReg` or :class:`WeakSIGReg`.
        lamb (float, optional): Trade-off weight for SIGReg.
            Default: ``0.02``
        normalize_projections (bool, optional): If ``True`` applies
            L2-normalization to projections before computing losses.
            Default: ``True``

    Shape:
        - Input: :math:`(V, N, D)` where :math:`V` is the number of views,
          :math:`N` is the batch size, and :math:`D` is the projection dim.
        - Output: Tuple ``(total_loss, invariance_loss, sigreg_loss)``

    Examples::

        >>> sigreg = SIGReg()
        >>> lejepa = LeJepaLoss(sigreg_loss_fn=sigreg)
        >>> proj = torch.randn(4, 32, 128, requires_grad=True)
        >>> total, inv, sig = lejepa(proj)
        >>> total.backward()
    """

    def __init__(
        self,
        sigreg_loss_fn: nn.Module,
        lamb: float = 0.02,
        normalize_projections: bool = True,
    ) -> None:
        super().__init__()
        self.sigreg_loss_fn = sigreg_loss_fn
        self.lamb = float(lamb)
        self.normalize_projections = normalize_projections

    def forward(
        self, proj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute the LeJEPA loss.

        Args:
            proj (torch.Tensor): Projected embeddings of shape
                :math:`(V, N, D)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ``(total_loss, invariance_loss, sigreg_loss)``
        """
        if proj.dim() != 3:
            raise ValueError(
                f"LeJepaLoss expects a 3D tensor of shape (V, N, D), "
                f"got shape {proj.shape}."
            )

        if self.normalize_projections:
            proj = torch.nn.functional.normalize(proj, dim=-1)

        # Invariance: each view should be close to the batch-mean projection
        proj_mean = proj.mean(dim=0, keepdim=True)
        invariance_loss = (proj_mean - proj).square().mean()

        # SIGReg expects (N, D) - flatten views
        V, N, D = proj.shape
        sigreg_in = proj.transpose(0, 1).reshape(N * V, D)
        sigreg_loss = self.sigreg_loss_fn(sigreg_in)

        total_loss = (
            (1.0 - self.lamb) * invariance_loss + self.lamb * sigreg_loss
        )

        return total_loss, invariance_loss, sigreg_loss


__all__ = [
    "LeJepaLoss",
    "LossName",
    "SIGReg",
    "WeakSIGReg",
]
