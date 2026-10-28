"""CP-Mamba: centre-prior guided selective scan (CP-SSM) on the stride-4 level.

A coarse class heatmap H_c on P3 (stride 8, focal loss on stride-8 CenterNet targets)
gives the prior S = MaxPool3x3(Up(max_c σ(H_c))) ∈ [0, 1] at P2. The P2 mixer uses S to

1. raise Δ: Δ_t = softplus(W_Δ x_t + b_Δ + α s_t), α ≥ 0 with α₀ = 0, on every route;
2. add a priority route that scans tokens from low to high prior (``bins`` levels,
   raster order inside a level), so candidates are read out last with most context;
3. restrict that route to the top-ρN prior tokens plus random extras while training;
   unselected tokens receive nothing from it and keep the block's identity residual.

Horizontal and vertical routes stay dense. The priority route is off for
``warmup_epochs`` and then faded in over ``ramp_epochs`` (``set_progress``).
``alpha_init`` > 0 starts the Δ modulation at a non-trivial strength (e^α at the prior
peaks); ``ramp_delta`` puts that modulation on the same warm-up and fade-in schedule.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .scan import (flatten_routes, gather_tokens, nonneg, priority_order, restore_routes,
                   scatter_tokens, select_tokens)
from .vss import SS2D, RouteParams, VSSBlock, plain_vss


@dataclass
class CPConfig:
    use_cp_block: bool = True      # False: plain VSS at P2, coarse heatmap as auxiliary task (B1)
    regular_routes: tuple = ('h', 'v')
    delta_mod: bool = True         # B2
    alpha_init: float = 0.0        # α₀; the design doc uses 0
    ramp_delta: bool = False       # scale α by the priority-route schedule (off in warm-up)
    priority: bool = True          # B3
    sparse_ratio: float = 0.15     # ρ of B4; 1.0 scans every token in priority order
    extra_ratio: float = 0.05      # random tokens added while training (B9: 0)
    bins: object = 4               # B7: quantisation levels, None = full sort
    descending: bool = False       # B6: high-to-low order
    dilate: bool = True            # B10: 3x3 max-pool dilation of the prior
    detach_prior: bool = False     # B12: stop the CP-SSM gradient into the coarse branch
    warmup_epochs: int = 5
    ramp_epochs: int = 5
    head_conv: int = 128           # channels of the coarse heatmap head

    def __post_init__(self):
        self.regular_routes = tuple(self.regular_routes)


def centre_prior(hm_c, size, dilate=True, detach=False):
    """Class-agnostic prior S (b, 1, h, w) in [0, 1] from coarse heatmap logits."""
    prior = hm_c.float().sigmoid().amax(dim=1, keepdim=True)
    if detach:
        prior = prior.detach()
    prior = F.interpolate(prior, size=size, mode='bilinear', align_corners=False)
    if dilate:
        prior = F.max_pool2d(prior, 3, stride=1, padding=1)
    return prior.clamp(0, 1)


class CoarseHeatmap(nn.Module):
    """CenterNet-style heatmap head on P3; its logits are returned as output 'hm_c'."""

    def __init__(self, cin, classes, head_conv=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(cin, head_conv, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, classes, 1, bias=True))
        self.head[-1].bias.data.fill_(-2.19)

    def forward(self, x, ctx):
        ctx.outputs['hm_c'] = self.head(x)


class CPSS2D(SS2D):
    """Dense regular routes plus the prior-ordered (sparse) priority route."""

    def __init__(self, d_model, cfg, expand=1.0, d_state=16, backend='auto'):
        super().__init__(d_model, expand=expand, d_state=d_state,
                         routes=cfg.regular_routes, backend=backend)
        self.cfg = cfg
        self.alpha = (nn.Parameter(torch.tensor(float(cfg.alpha_init))) if cfg.delta_mod
                      else None)
        self.prio = RouteParams(1, self.d_inner, d_state, self.dt_rank) if cfg.priority else None
        self.register_buffer('prio_gate', torch.tensor(float(cfg.priority)))
        self._gate = float(cfg.priority)
        self.register_load_state_dict_post_hook(_sync_gate)

    def set_progress(self, completed_epochs):
        cfg = self.cfg
        if self.prio is None and not cfg.ramp_delta:
            return
        done = completed_epochs - cfg.warmup_epochs
        gate = 0.0 if done < 0 else min(1.0, (done + 1) / max(1, cfg.ramp_epochs))
        self.prio_gate.fill_(gate)
        self._gate = gate

    def context_inputs(self, ctx):
        return (ctx.outputs['hm_c'],)

    def mix(self, xi, hm_c):
        cfg = self.cfg
        b, d, h, w = xi.shape
        prior = centre_prior(hm_c, (h, w), cfg.dilate, cfg.detach_prior)
        alpha = nonneg(self.alpha) if self.alpha is not None else None
        if alpha is not None and cfg.ramp_delta:
            alpha = alpha * self._gate
        xs = flatten_routes(xi, self.routes)
        dt, B, C = self.ssm.project(xs)
        if alpha is not None:
            dt = dt + alpha * flatten_routes(prior, self.routes)
        y = restore_routes(self.ssm.scan(xs, dt, B, C, backend=self.backend),
                           self.routes, h, w).sum(1)
        if self.prio is None or self._gate <= 0:
            return (y,)
        score = prior.flatten(1)
        if cfg.sparse_ratio < 1:
            extra = cfg.extra_ratio if self.training else 0.0
            idx = select_tokens(score, cfg.sparse_ratio, extra, cfg.bins, cfg.descending)
        else:
            idx = priority_order(score, cfg.bins, cfg.descending)
        xp = gather_tokens(xi.flatten(2), idx).unsqueeze(1)
        dtp, Bp, Cp = self.prio.project(xp)
        if alpha is not None:
            dtp = dtp + alpha * score.gather(1, idx)[:, None, None, :]
        yp = self.prio.scan(xp, dtp, Bp, Cp, backend=self.backend)[:, 0]
        return (y + self._gate * scatter_tokens(yp, idx, h * w).view(b, d, h, w),)


def _sync_gate(module, incompatible_keys):
    module._gate = float(module.prio_gate)


def cp_block(dim, expand, cfg, d_state=16, backend='auto', use_checkpoint=False, **block_kw):
    if not cfg.use_cp_block:
        return plain_vss(dim, expand, d_state, backend, use_checkpoint, **block_kw)
    return VSSBlock(dim, CPSS2D(dim, cfg, expand=expand, d_state=d_state, backend=backend),
                    use_checkpoint, **block_kw)
