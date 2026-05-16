from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDiffusion(nn.Module):
    """Forward noising and reverse DDPM/DDIM sampling for Raman spectra."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) using the closed-form DDPM equation."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self.sqrt_alphas_cumprod[t][:, None] * x0
            + self.sqrt_one_minus_alphas_cumprod[t][:, None] * noise
        )

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    @torch.no_grad()
    def p_sample_ddpm(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        label: torch.Tensor,
    ) -> torch.Tensor:
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_pred = model(x_t, t_batch, label)
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        mean = self.sqrt_recip_alphas[t] * (x_t - beta_t / sqrt_one_minus_alpha_bar * eps_pred)
        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(self.posterior_variance[t]) * noise

    @torch.no_grad()
    def p_sample_ddim(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        label: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_pred = model(x_t, t_batch, label)
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod_prev[t]
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        if t == 0:
            return x0_pred

        sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev))
        direction = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * eps_pred
        noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
        return torch.sqrt(alpha_prev) * x0_pred + direction + noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int],
        label: torch.Tensor,
        device: torch.device,
        sampler: str = "ddim",
        eta: float = 0.0,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            if sampler == "ddpm":
                x = self.p_sample_ddpm(model, x, t, label)
            elif sampler == "ddim":
                x = self.p_sample_ddim(model, x, t, label, eta=eta)
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
        return x

    def compute_loss(self, model: nn.Module, x0: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Train the network to predict the Gaussian noise added at time t."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, t, label)
        return F.mse_loss(eps_pred, noise)
