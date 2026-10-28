"""Zero-initialised Mamba add-ons on CenterNet's own DLA-34 neck (DCN kept).

CenterNet's DLASeg is used unchanged: DLA-34 -> DLAUp -> IDAUp (16 DCNv2 layers) -> heads,
with the same parameter names, so its checkpoints load directly. Residual blocks are
inserted at two points of its forward pass:

    stride 8  on the DLAUp P3 map (128 ch), before the final IDAUp folds it into P2;
    stride 4  on the IDAUp output (64 ch) that the heads read.

Every block ends in zero-initialised projections and the innovation input of the heatmap
head starts at zero, so at initialisation the detector computes exactly the DCN baseline
and any change in AP belongs to the add-on. ``new_parameters`` lists what the baseline
does not have (trained with ``train.new_lr_mult``).

Variants
    dlavss  plain VSS blocks at both points (control of both ideas, A1')
    dlacp   VSS at stride 8, coarse heatmap on the stride-8 map, CP-SSM at stride 4
            (B5'; B1' with cp.use_cp_block false)
    dlaia   IA-VSS at both points, innovation maps into the heatmap head (A4')
"""
import torch
import torch.nn.functional as F
from torch import nn

from .cp_mamba import CoarseHeatmap, CPConfig, CPSS2D, cp_block
from .fpn import NeckContext
from .ia_mamba import IAConfig, ia_block
from .vss import plain_vss

ADDON_DEFAULTS = dict(depth8=2, depth4=2, expand8=2, expand4=1, mlp_ratio=4.0, d_state=16,
                      backend='auto', checkpoint=False, micro_batch=0)
ADDON_VARIANTS = ('dlavss', 'dlacp', 'dlaia')
STRIDE4, STRIDE8 = 0, 1    # NeckContext levels, numbered like Mamba-FPN (0 = stride 4)


class DLAAddon(nn.Module):

    def __init__(self, heads, head_conv, variant='dlavss', addon=None, cp=None, ia=None,
                 pretrained=True):
        super().__init__()
        from models.networks.pose_dla_dcn import DLASeg  # CenterNet

        if variant not in ADDON_VARIANTS:
            raise ValueError('unknown variant %s' % variant)
        if head_conv <= 0:
            raise ValueError('CenterNet DLA heads use head_conv > 0')
        seg = DLASeg('dla34', heads, pretrained=pretrained, down_ratio=4, final_kernel=1,
                     last_level=5, head_conv=head_conv)
        self.first_level, self.last_level = seg.first_level, seg.last_level
        self.base, self.dla_up, self.ida_up = seg.base, seg.dla_up, seg.ida_up
        self.variant = variant
        self.heads = dict(heads)
        for head in self.heads:
            self.__setattr__(head, getattr(seg, head))

        acfg = {**ADDON_DEFAULTS, **(addon or {})}
        ch4, ch8 = self.base.channels[self.first_level], self.base.channels[self.first_level + 1]
        block_kw = dict(d_state=acfg['d_state'], backend=acfg['backend'],
                        use_checkpoint=bool(acfg['checkpoint']),
                        mlp_ratio=float(acfg['mlp_ratio']), zero_init=True,
                        micro_batch=int(acfg['micro_batch']))
        self.cp_cfg = self.ia_cfg = None
        self.coarse = self.hm_innov = None
        if variant == 'dlavss':
            def make_block(level, dim, expand):
                return plain_vss(dim, expand, **block_kw)
        elif variant == 'dlacp':
            self.cp_cfg = CPConfig(**(cp or {}))
            self.coarse = CoarseHeatmap(ch8, self.heads['hm'], self.cp_cfg.head_conv)

            def make_block(level, dim, expand):
                if level == STRIDE4:
                    return cp_block(dim, expand, self.cp_cfg, **block_kw)
                return plain_vss(dim, expand, **block_kw)
        else:
            self.ia_cfg = IAConfig(**(ia or {}))

            def make_block(level, dim, expand):
                return ia_block(dim, expand, level, self.ia_cfg, **block_kw)
        self.stage8 = nn.ModuleList(make_block(STRIDE8, ch8, acfg['expand8'])
                                    for _ in range(acfg['depth8']))
        self.stage4 = nn.ModuleList(make_block(STRIDE4, ch4, acfg['expand4'])
                                    for _ in range(acfg['depth4']))
        if self.ia_cfg is not None and self.ia_cfg.to_head:
            depth = {STRIDE4: acfg['depth4'], STRIDE8: acfg['depth8']}
            if any(depth.get(level, 0) == 0 for level in self.ia_cfg.head_levels):
                raise ValueError('ia.head_levels must name stages that have blocks (0, 1)')
            # A zero-initialised conv over the innovation maps added to the first head conv
            # equals concatenating them as extra input channels with zero weights.
            self.hm_innov = nn.Conv2d(len(self.ia_cfg.head_levels), self.hm[0].out_channels,
                                      3, padding=1, bias=False)
            nn.init.zeros_(self.hm_innov.weight)

    def new_parameters(self):
        """Parameters the DCN baseline does not have."""
        modules = (self.stage8, self.stage4, self.coarse, self.hm_innov)
        return [p for m in modules if m is not None for p in m.parameters()]

    def set_epoch(self, completed_epochs):
        """Called by the trainer before each epoch (CP-SSM warm-up and fade-in)."""
        for module in self.modules():
            if isinstance(module, CPSS2D):
                module.set_progress(completed_epochs)

    def forward(self, x):
        ctx = NeckContext()
        x = self.dla_up(self.base(x))
        y = [x[i].clone() for i in range(self.last_level - self.first_level)]
        ctx.level = STRIDE8
        for block in self.stage8:
            y[1] = block(y[1], ctx)
        if self.coarse is not None:
            self.coarse(y[1], ctx)
        self.ida_up(y, 0, len(y))
        p2 = y[-1]
        ctx.level = STRIDE4
        for block in self.stage4:
            p2 = block(p2, ctx)
        ret = {}
        for head in self.heads:
            if head == 'hm' and self.hm_innov is not None:
                maps = [ctx.innovation_maps[level] for level in self.ia_cfg.head_levels]
                maps = [m if m.shape[-2:] == p2.shape[-2:] else
                        F.interpolate(m, size=p2.shape[-2:], mode='bilinear', align_corners=False)
                        for m in maps]
                hidden = self.hm[0](p2) + self.hm_innov(torch.cat(maps, dim=1).to(p2.dtype))
                ret[head] = self.hm[2](self.hm[1](hidden))
            else:
                ret[head] = getattr(self, head)(p2)
        ret.update(ctx.outputs)
        if self.training and ctx.aux_losses:
            ret['pred_loss'] = torch.stack(ctx.aux_losses).mean()
        return [ret]
