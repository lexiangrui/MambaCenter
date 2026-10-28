"""CenterNet detectors with a Mamba neck.

DLA-34 base (CenterNet's implementation and ImageNet weights) -> Mamba-FPN -> CenterNet
heatmap / size / offset heads at stride 4, built exactly like CenterNet's DLASeg heads.

Variants
    mambafpn  plain VSS blocks at every level (direct baseline of both ideas, A1)
    cpmamba   VSS at P5–P3, coarse heatmap on P3, CP-SSM at P2 (CP-Mamba)
    iamamba   IA-VSS at every level, innovation maps into the heatmap head (IA-Mamba)
"""
import torch
import torch.nn.functional as F
from torch import nn

from .cp_mamba import CoarseHeatmap, CPConfig, CPSS2D, cp_block
from .fpn import MambaFPN, NeckContext
from .ia_mamba import IAConfig, ia_block
from .vss import plain_vss

NECK_DEFAULTS = dict(dims=(64, 128, 128, 128), depths=(1, 2, 2, 2), expands=(1, 2, 2, 2),
                     d_state=16, backend='auto', checkpoint=False)
VARIANTS = ('mambafpn', 'cpmamba', 'iamamba')


class MambaCenterNet(nn.Module):

    def __init__(self, heads, head_conv, variant='mambafpn', neck=None, cp=None, ia=None,
                 pretrained=True):
        super().__init__()
        from models.networks.pose_dla_dcn import dla34, fill_fc_weights  # CenterNet

        if variant not in VARIANTS:
            raise ValueError('unknown variant %s' % variant)
        if head_conv <= 0:
            raise ValueError('CenterNet DLA heads use head_conv > 0')
        self.variant = variant
        self.heads = dict(heads)
        self.base = dla34(pretrained=pretrained)
        ncfg = {**NECK_DEFAULTS, **(neck or {})}
        dims = list(ncfg['dims'])
        block_kw = dict(d_state=ncfg['d_state'], backend=ncfg['backend'],
                        use_checkpoint=bool(ncfg['checkpoint']))
        self.cp_cfg = self.ia_cfg = None
        hooks = {}
        if variant == 'mambafpn':
            def make_block(level, index, dim, expand):
                return plain_vss(dim, expand, **block_kw)
        elif variant == 'cpmamba':
            self.cp_cfg = CPConfig(**(cp or {}))
            self.coarse = CoarseHeatmap(dims[1], self.heads['hm'], self.cp_cfg.head_conv)
            hooks[1] = self.coarse

            def make_block(level, index, dim, expand):
                if level == 0:
                    return cp_block(dim, expand, self.cp_cfg, **block_kw)
                return plain_vss(dim, expand, **block_kw)
        else:
            self.ia_cfg = IAConfig(**(ia or {}))

            def make_block(level, index, dim, expand):
                return ia_block(dim, expand, level, self.ia_cfg, **block_kw)
        self.neck = MambaFPN(self.base.channels[2:6], dims, ncfg['depths'], ncfg['expands'],
                             make_block, hooks)
        self.neck_out = nn.Sequential(nn.BatchNorm2d(dims[0]), nn.ReLU(inplace=True))
        self.extra_hm_channels = (len(self.ia_cfg.head_levels)
                                  if self.ia_cfg is not None and self.ia_cfg.to_head else 0)
        for head, classes in self.heads.items():
            cin = dims[0] + (self.extra_hm_channels if head == 'hm' else 0)
            fc = nn.Sequential(
                nn.Conv2d(cin, head_conv, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes, 1, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            if head == 'hm' and self.extra_hm_channels:
                # Innovation inputs start silent: the head equals the plain one at init.
                fc[0].weight.data[:, dims[0]:].zero_()
            self.__setattr__(head, fc)

    def set_epoch(self, completed_epochs):
        """Called by the trainer before each epoch (CP-Mamba priority-route schedule)."""
        for module in self.modules():
            if isinstance(module, CPSS2D):
                module.set_progress(completed_epochs)

    def forward(self, x):
        ctx = NeckContext()
        p2 = self.neck_out(self.neck(self.base(x)[2:6], ctx)[0])
        hm_input = p2
        if self.extra_hm_channels:
            maps = [ctx.innovation_maps[level] for level in self.ia_cfg.head_levels]
            maps = [m if m.shape[-2:] == p2.shape[-2:] else
                    F.interpolate(m, size=p2.shape[-2:], mode='bilinear', align_corners=False)
                    for m in maps]
            hm_input = torch.cat([p2] + [m.to(p2.dtype) for m in maps], dim=1)
        ret = {head: getattr(self, head)(hm_input if head == 'hm' else p2) for head in self.heads}
        ret.update(ctx.outputs)
        if self.training and ctx.aux_losses:
            ret['pred_loss'] = torch.stack(ctx.aux_losses).mean()
        return [ret]
