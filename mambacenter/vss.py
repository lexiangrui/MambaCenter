"""Plain VSS block: VMamba-style SS2D over raster routes, the building block of Mamba-FPN.

SS2D: LayerNorm'ed tokens -> in_proj (x, z) -> depthwise 3x3 + SiLU -> per-route x/Δ
projections -> native selective scan -> restore and sum routes -> LayerNorm · SiLU(z)
-> out_proj. Projections, Δ and the scan stay in fp32 under autocast.

Mixers are pure functions of their tensor inputs: ``context_inputs`` reads what a mixer
needs from the neck context, ``forward(x, *inputs)`` returns ``(y, *extras)`` and
``write_context`` stores the extras. This lets VSSBlock recompute a block in backward
(activation checkpointing) without seeing context written by later blocks.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .scan import ROUTES, flatten_routes, restore_routes, selective_scan


class RouteParams(nn.Module):
    """x/Δ projections and SSM parameters (A, D, Δ bias) of ``k`` routes over ``d`` channels.

    Initialisation follows Mamba: A = -(1..N), D = 1, Δ bias = softplus⁻¹(Δ0) with
    Δ0 log-uniform in [dt_min, dt_max].
    """

    def __init__(self, k, d_inner, d_state, dt_rank, dt_min=1e-3, dt_max=0.1, dt_floor=1e-4):
        super().__init__()
        self.k, self.d_inner, self.d_state, self.dt_rank = k, d_inner, d_state, dt_rank
        bound = d_inner ** -0.5
        self.x_proj = nn.Parameter(
            torch.empty(k, dt_rank + 2 * d_state, d_inner).uniform_(-bound, bound))
        self.dt_proj = nn.Parameter(
            torch.empty(k, d_inner, dt_rank).uniform_(-dt_rank ** -0.5, dt_rank ** -0.5))
        dt = torch.exp(torch.rand(k, d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(
            torch.arange(1, d_state + 1, dtype=torch.float32).log().repeat(k * d_inner, 1))
        self.D = nn.Parameter(torch.ones(k * d_inner))

    def project(self, xs):
        """xs (b, k, d, l) -> Δ pre-activation without bias (b, k, d, l), B and C (b, k, n, l)."""
        w_dt, w_B, w_C = self.x_proj.split([self.dt_rank, self.d_state, self.d_state], dim=1)
        dt = torch.einsum('bkdl,krd->bkrl', xs, w_dt)
        B = torch.einsum('bkdl,knd->bknl', xs, w_B)
        C = torch.einsum('bkdl,knd->bknl', xs, w_C)
        return torch.einsum('bkrl,kdr->bkdl', dt, self.dt_proj), B, C

    def scan(self, xs, delta, B, C, softplus=True, backend='auto'):
        """softplus=True: Δ = softplus(delta + bias) in the kernel; False: ``delta`` is Δ itself."""
        b, k, d, length = xs.shape
        y = selective_scan(
            xs.reshape(b, k * d, length), delta.reshape(b, k * d, length),
            -torch.exp(self.A_log.float()), B, C, self.D.float(),
            delta_bias=self.dt_bias.float().reshape(-1) if softplus else None,
            delta_softplus=softplus, backend=backend)
        return y.view(b, k, d, length)

    def delta(self, dt):
        """Explicit Δ = softplus(dt + bias) for mixers that modulate Δ after the softplus."""
        return F.softplus(dt + self.dt_bias.float()[None, :, :, None])


class SS2D(nn.Module):
    """2D selective scan over ``routes``; subclasses change how routes are scanned in ``mix``."""

    def __init__(self, d_model, expand=2.0, d_state=16, routes=ROUTES, dt_rank='auto',
                 d_conv=3, backend='auto'):
        super().__init__()
        self.d_inner = d_inner = int(round(expand * d_model))
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else int(dt_rank)
        self.d_state = d_state
        self.routes = tuple(routes)
        self.backend = backend
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv2d = nn.Conv2d(d_inner, d_inner, d_conv, padding=d_conv // 2, groups=d_inner)
        self.ssm = RouteParams(len(self.routes), d_inner, d_state, self.dt_rank)
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def context_inputs(self, ctx):
        return ()

    def write_context(self, ctx, extras):
        pass

    def forward(self, x, *inputs):
        """x (b, h, w, c) normalised tokens -> (y (b, h, w, c), *extras)."""
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = F.silu(self.conv2d(xi.permute(0, 3, 1, 2).contiguous()))
        with torch.autocast(device_type=xi.device.type, enabled=False):
            y, *extras = self.mix(xi.float(), *inputs)
            y = self.out_norm(y.permute(0, 2, 3, 1))
        return (self.out_proj(y * F.silu(z)), *extras)

    def mix(self, xi):
        """xi (b, d, h, w) fp32 -> (summed route outputs (b, d, h, w) fp32,)."""
        h, w = xi.shape[-2:]
        xs = flatten_routes(xi, self.routes)
        dt, B, C = self.ssm.project(xs)
        y = self.ssm.scan(xs, dt, B, C, backend=self.backend)
        return (restore_routes(y, self.routes, h, w).sum(1),)


class VSSBlock(nn.Module):
    """x + mixer(LayerNorm(x)) on (b, c, h, w) maps, optionally recomputed in backward.

    ``mlp_ratio`` > 0 adds VMamba's feed-forward sub-block x + MLP(LayerNorm(x));
    ``zero_init`` zeroes the last projection of every branch so the block starts as the
    identity (residual add-ons on a trained neck). ``micro_batch`` > 0 runs (and
    recomputes) the block on that many images at a time: every operation inside a block is
    per image, so outputs and gradients are unchanged while the peak memory shrinks.
    """

    def __init__(self, dim, mixer, use_checkpoint=False, mlp_ratio=0.0, zero_init=False,
                 micro_batch=0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = mixer
        self.use_checkpoint = use_checkpoint
        self.micro_batch = int(micro_batch)
        self.mlp = None
        if mlp_ratio > 0:
            hidden = int(round(dim * mlp_ratio))
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        if zero_init:
            nn.init.zeros_(mixer.out_proj.weight)
            if self.mlp is not None:
                nn.init.zeros_(self.mlp[-1].weight)
                nn.init.zeros_(self.mlp[-1].bias)

    def _run(self, x, *inputs):
        y, *extras = self.mixer(self.norm(x.permute(0, 2, 3, 1)), *inputs)
        x = x + y.permute(0, 3, 1, 2)
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return (x, *extras)

    def _call(self, x, *inputs):
        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(self._run, x, *inputs, use_reentrant=False)
        return self._run(x, *inputs)

    def forward(self, x, ctx=None):
        inputs = self.mixer.context_inputs(ctx)
        n = x.shape[0]
        if not 0 < self.micro_batch < n:
            x, *extras = self._call(x, *inputs)
        else:
            parts = [self._call(x[i:i + self.micro_batch],
                                *[t[i:i + self.micro_batch] if torch.is_tensor(t) else t
                                  for t in inputs])
                     for i in range(0, n, self.micro_batch)]
            sizes = [p[0].shape[0] for p in parts]
            x = torch.cat([p[0] for p in parts])
            # Per-image extras are concatenated, scalar ones (IA L_pred, a mean over equally
            # sized images) are averaged with image-count weights.
            extras = [torch.cat(vals) if vals[0].dim() else
                      sum(v * s for v, s in zip(vals, sizes)) / n
                      for vals in zip(*[p[1:] for p in parts])]
        self.mixer.write_context(ctx, extras)
        return x


def plain_vss(dim, expand=2.0, d_state=16, backend='auto', use_checkpoint=False, **block_kw):
    return VSSBlock(dim, SS2D(dim, expand=expand, d_state=d_state, backend=backend),
                    use_checkpoint, **block_kw)
