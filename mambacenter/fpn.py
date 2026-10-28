"""Mamba-FPN: top-down VSS neck from DLA-34 C2–C5 to the stride-4 map used by the heads."""
import torch.nn.functional as F
from torch import nn


class NeckContext:
    """State shared by the blocks of one forward pass."""

    def __init__(self):
        self.level = None            # pyramid index being processed (0 = P2, stride 4)
        self.innovation = None       # IA-Mamba: (level, E) of the latest block
        self.innovation_maps = {}    # IA-Mamba: level -> E of the level's last block
        self.aux_losses = []         # scalar losses produced inside the neck (IA L_pred)
        self.outputs = {}            # extra detector outputs, e.g. CP coarse heatmap 'hm_c'


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))


class MambaFPN(nn.Module):
    """P5 = blocks(L5(C5)); P_l = blocks(L_l(C_l) + Up(P_{l+1})) for l = 4, 3, 2.

    ``make_block(level, index, dim, expand)`` builds each block (level 0 = P2) and
    ``hooks[level](feature, ctx)`` runs after a level (CP-Mamba: coarse heatmap on P3).
    """

    def __init__(self, in_channels, dims, depths, expands, make_block, hooks=None):
        super().__init__()
        n = len(in_channels)
        if not len(dims) == len(depths) == len(expands) == n:
            raise ValueError('dims, depths and expands need one entry per pyramid level')
        self.lateral = nn.ModuleList(conv_bn(c, d) for c, d in zip(in_channels, dims))
        self.top_down = nn.ModuleList(
            conv_bn(dims[i + 1], dims[i]) if dims[i + 1] != dims[i] else nn.Identity()
            for i in range(n - 1))
        self.levels = nn.ModuleList(
            nn.ModuleList(make_block(i, j, dims[i], expands[i]) for j in range(depths[i]))
            for i in range(n))
        self.hooks = dict(hooks or {})

    def forward(self, feats, ctx):
        outs = [None] * len(feats)
        top = None
        for i in reversed(range(len(feats))):
            x = self.lateral[i](feats[i])
            if top is not None:
                x = x + F.interpolate(self.top_down[i](top), size=x.shape[-2:],
                                      mode='bilinear', align_corners=False)
            ctx.level = i
            for block in self.levels[i]:
                x = block(x, ctx)
            if i in self.hooks:
                self.hooks[i](x, ctx)
            outs[i] = top = x
        return outs
