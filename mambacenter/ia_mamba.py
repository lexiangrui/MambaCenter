"""IA-Mamba: innovation-driven selective scan (IA-SSM) in every neck block.

Each IA-VSS block predicts its scan input from the previous scan output of the same
route, x̂_t = P y_{t-1} (no kernel change: the output is shifted by one token), and
measures the innovation e_t = x_t − x̂_t. Per route s = ‖e‖₂ / (AvgPool_k(‖e‖₂) + ε),
and the four routes are fused by a soft minimum E = −τ log Σ_d exp(−s_d / τ): an edge
is predictable along its own direction, a point target along none.

E of block l raises Δ of block l+1 of the same level with a lower bound,
Δ = softplus(W_Δ x + b) + γ softplus(aE + c), γ ≥ 0, γ₀ = 0, so a failing signal
degrades to the plain VSS. The last E of P2–P4 is upsampled and concatenated to the
heatmap-head input, and L_pred = mean (sg(x_t) − x̂_t)² trains P to predict background.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .scan import ROUTES, flatten_routes, nonneg, restore_routes, soft_min
from .vss import SS2D, VSSBlock


@dataclass
class IAConfig:
    kernel: int = 7                # local normalisation window k
    tau: float = 0.5               # soft-min temperature τ
    eps: float = 1e-6
    fusion: str = 'softmin'        # A7: 'mean' | 'max'
    local_norm: bool = True        # A8
    predictor: str = 'linear'      # P: 'linear' (d x d) or 'diag' (per channel)
    modulate: bool = True          # A2: False keeps Δ unmodulated
    to_head: bool = True           # A3: False keeps E out of the heatmap head
    lower_bound: bool = True       # A5: False lets the modulation lower Δ
    target: str = 'delta'          # A6: 'B' scales only the write gain B̄
    cross_level: bool = False      # also let a level's first block use the previous level's E
    head_levels: tuple = (0, 1, 2)  # pyramid levels whose E feeds the heatmap head (P2–P4)

    def __post_init__(self):
        self.head_levels = tuple(self.head_levels)
        if self.fusion not in ('softmin', 'mean', 'max'):
            raise ValueError('fusion must be softmin, mean or max')
        if self.target not in ('delta', 'B'):
            raise ValueError("target must be 'delta' or 'B'")


class IASS2D(SS2D):
    """Four-route SS2D that emits the innovation map E and consumes the previous one."""

    def __init__(self, d_model, level, cfg, expand=2.0, d_state=16, backend='auto'):
        super().__init__(d_model, expand=expand, d_state=d_state, routes=ROUTES, backend=backend)
        self.level, self.cfg = level, cfg
        d = self.d_inner
        # P starts at zero: x̂ = 0, so E begins as a locally normalised feature contrast.
        self.P = nn.Parameter(torch.zeros(d, d) if cfg.predictor == 'linear' else torch.zeros(d))
        self.gamma = nn.Parameter(torch.zeros(()))
        self.a = nn.Parameter(torch.ones(()))
        self.c = nn.Parameter(torch.zeros(()))

    def context_inputs(self, ctx):
        """E of the previous block of this level (or of the previous level if cross_level)."""
        if not self.cfg.modulate or ctx.innovation is None:
            return (None,)
        level, E = ctx.innovation
        return (E if level == self.level or self.cfg.cross_level else None,)

    def write_context(self, ctx, extras):
        E, pred = extras
        ctx.innovation = (self.level, E)
        ctx.innovation_maps[self.level] = E
        if self.training:
            ctx.aux_losses.append(pred)

    def mix(self, xi, E_in):
        cfg = self.cfg
        h, w = xi.shape[-2:]
        xs = flatten_routes(xi, self.routes)
        dt, B, C = self.ssm.project(xs)
        if E_in is not None and E_in.shape[-2:] != (h, w):
            E_in = F.interpolate(E_in, size=(h, w), mode='bilinear', align_corners=False)
        if E_in is None:
            y = self.ssm.scan(xs, dt, B, C, backend=self.backend)
        else:
            gamma = nonneg(self.gamma)
            e = flatten_routes(E_in.float(), self.routes)
            if cfg.target == 'B':
                y = self.ssm.scan(xs, dt, B * (1 + gamma * F.softplus(self.a * e + self.c)), C,
                                  backend=self.backend)
            elif cfg.lower_bound:
                delta = self.ssm.delta(dt) + gamma * F.softplus(self.a * e + self.c)
                y = self.ssm.scan(xs, delta, B, C, softplus=False, backend=self.backend)
            else:
                y = self.ssm.scan(xs, dt + gamma * (self.a * e + self.c), B, C,
                                  backend=self.backend)
        E, pred = self.innovation(xs, y, h, w)
        return restore_routes(y, self.routes, h, w).sum(1), E, pred

    def innovation(self, xs, y, h, w):
        """xs, y (b, k, d, l) -> E (b, 1, h, w) and the prediction loss."""
        cfg = self.cfg
        y_prev = F.pad(y[..., :-1], (1, 0))
        if self.P.dim() == 2:
            x_hat = torch.einsum('od,bkdl->bkol', self.P, y_prev)
        else:
            x_hat = self.P[None, None, :, None] * y_prev
        pred = (xs.detach() - x_hat).pow(2).mean()
        norm = (xs - x_hat).pow(2).sum(2, keepdim=True).add(1e-12).sqrt()
        s = restore_routes(norm, self.routes, h, w)[:, :, 0]
        if cfg.local_norm:
            k = cfg.kernel
            s = s / (F.avg_pool2d(s, k, stride=1, padding=k // 2, count_include_pad=False) + cfg.eps)
        if cfg.fusion == 'softmin':
            E = soft_min(s, cfg.tau, dim=1)
        elif cfg.fusion == 'mean':
            E = s.mean(1)
        else:
            E = s.amax(1)
        return E.unsqueeze(1), pred


def ia_block(dim, expand, level, cfg, d_state=16, backend='auto', use_checkpoint=False,
             **block_kw):
    return VSSBlock(dim, IASS2D(dim, level, cfg, expand=expand, d_state=d_state, backend=backend),
                    use_checkpoint, **block_kw)
