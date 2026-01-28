"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional

from src.models.nn import DropoutNd
from src.models.sequence.kernels.kernel import Kernel
print("[INFO] Using FRACTIONAL S4D")
def torch_mittag_leffler(z, alpha, beta, series_terms, max_abs=50.0):
    if not z.is_complex():
        z = z.to(torch.complex64)

    abs_z = torch.abs(z)
    z = torch.where(abs_z > max_abs, z / abs_z * max_abs, z)

    k = torch.arange(series_terms, device=z.device, dtype=z.real.dtype)
    gamma = torch.exp(torch.lgamma(alpha * k + beta))

    term = torch.ones_like(z)
    out = term / gamma[0]

    for i in range(1, series_terms):
        term = term * z
        out = out + term / gamma[i]

    return out

class FractionalS4DKernel(Kernel):
    """Generate convolution kernel from fractional diagonal SSM parameters."""

    def __init__(
        self,
        d_model,
        channels=1,
        l_max=None,
        d_state=64,
        dt_min=0.001,
        dt_max=0.1,
        alpha=0.25,
        beta=1.0,
        ml_terms=16,
        lr=None,
        wd=0.0,
        verbose=True,
        **kwargs,
    ):
        # Initialize Kernel base class
        super().__init__(
            d_model=d_model,
            channels=channels,
            l_max=l_max,
            lr=lr,
            wd=wd,
            verbose=verbose,
            **kwargs,
        )

        self.N = d_state // 2
        # Handle None values from config (Hydra null)
        self.alpha = alpha if alpha is not None else 0.25
        if self.verbose:
            print("[Kernel] alpha =", self.alpha)
        self.beta = beta if beta is not None else 1.0
        self.ml_terms = ml_terms if ml_terms is not None else 16
        self.dt_min = dt_min
        self.dt_max = dt_max

        # dt
        log_dt = torch.rand(self.H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # C - expand for channels
        C = 1e-5 * torch.randn(self.channels, self.H, self.N, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, self.lr_dict.get('dt', lr), self.wd_dict.get('dt', wd))

        # A spectrum
        log_A_real = torch.log(0.5 * torch.ones(self.H, self.N))
        A_imag = math.pi * repeat(torch.arange(self.N), 'n -> h n', h=self.H)
        self.register("log_A_real", log_A_real, self.lr_dict.get('A', lr), self.wd_dict.get('A', wd))
        self.register("A_imag", A_imag, self.lr_dict.get('A', lr), self.wd_dict.get('A', wd))

    def forward(self, state=None, rate=1.0, L=None):
        """Generate fractional S4D convolution kernel.
        
        Args:
            state: Not used (for compatibility with Kernel interface)
            rate: Sampling rate multiplier (not currently used)
            L: Kernel length
            
        Returns:
            K: (C, H, L) convolution kernel
            K_state: None (no state forwarding support yet)
        """
        if L is None:
            L = self.L if self.L is not None else 1
        
        dt = torch.exp(self.log_dt)                     # (H)
        C = torch.view_as_complex(self.C)               # (C, H, N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N)

        # Special case: alpha=1, beta=1 reduces to standard S4D ZOH kernel
        # E_{1,1}(z) = exp(z), so we can use the standard formula for efficiency
        if abs(self.alpha - 1.0) < 1e-6 and abs(self.beta - 1.0) < 1e-6:
            # Standard S4D ZOH kernel: K = log_vandermonde(C * (exp(dtA) - 1) / A, dtA, L)
            dtA = dt.unsqueeze(-1) * A  # (H, N)
            C_scaled = C * (torch.exp(dtA).unsqueeze(0) - 1.0) / A.unsqueeze(0)  # (C, H, N)
            
            # Compute log_vandermonde equivalent: 2 * Re(Σ C_scaled * exp(dtA * ell))
            ell = torch.arange(L, device=A.device, dtype=A.real.dtype)  # (L) - use real dtype, not complex
            exp_dtA_ell = torch.exp(dtA.unsqueeze(-1) * ell.unsqueeze(0).unsqueeze(0))  # (H, N, L)
            K = 2 * torch.real(
                torch.sum(C_scaled.unsqueeze(-1) * exp_dtA_ell.unsqueeze(0), dim=2)
            )  # (C, H, L)
        else:
            # Fractional case: use Mittag-Leffler function
            ell = torch.arange(L, device=A.device)

            # fractional time grid
            t1 = ((ell + 1)[None, :] * dt[:, None]) ** self.alpha
            t0 = (ell[None, :] * dt[:, None]) ** self.alpha

            Z1 = A.unsqueeze(-1) * t1.unsqueeze(1)  # (H, N, L)
            Z0 = A.unsqueeze(-1) * t0.unsqueeze(1)  # (H, N, L)

            E1 = torch_mittag_leffler(Z1, self.alpha, self.beta, self.ml_terms)
            E0 = torch_mittag_leffler(Z0, self.alpha, self.beta, self.ml_terms)

            # fractional ZOH kernel
            # C: (C, H, N), E1-E0: (H, N, L), A: (H, N)
            K = 2 * torch.real(
                torch.sum(C.unsqueeze(-1) * (E1 - E0).unsqueeze(0) / A.unsqueeze(-1).unsqueeze(0), dim=2)
            )  # (C, H, L)
        
        if self.verbose and L <= 10:  # Only print for short kernels
            print("Kernel min/max:", K.min().item(), K.max().item())

        return K, None  # Return (kernel, state_info) as per Kernel interface


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = FractionalS4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        # Note: rfft output shape is (..., L+1) for n=2*L
        k_f = torch.fft.rfft(k, n=2*L) # (H, L+1) [complex]
        u_f = torch.fft.rfft(u, n=2*L) # (B, H, L+1) [complex]
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B, H, L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None 
        
# -------------------------------------------------
# Quick sanity test
# -------------------------------------------------
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    for alpha in [1.0, 0.8, 0.6, 0.4, 0.25]:
        print("\nTesting alpha =", alpha)

        model = S4D(
            d_model=8,
            d_state=64,
            alpha=alpha,
            beta=1.0,
            ml_terms=32,
        )

        x = torch.randn(2, 8, 256)
        y, _ = model(x)

        print("output shape:", y.shape)
        print("mean:", y.mean().item())
        print("std :", y.std().item())
