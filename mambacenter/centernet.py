"""Registration of the Mamba detectors as CenterNet architectures.

``register(model_cfg)`` adds ``mambafpn_34``, ``cpmamba_34`` and ``iamamba_34`` (Mamba
necks) and ``dlavss_34``, ``dlacp_34`` and ``dlaia_34`` (add-ons on CenterNet's DCN
neck) to CenterNet's model factory, so CenterNet's create_model / load_model / detectors
build them like ``dla_34``. CenterNet's ``src/lib`` must be importable.
"""
from functools import partial

from .addon import ADDON_VARIANTS, DLAAddon
from .detector import VARIANTS, MambaCenterNet


def build(variant, num_layers, heads, head_conv, cfg=None):
    if num_layers != 34:
        raise ValueError('only the DLA-34 backbone is implemented (arch %s_34)' % variant)
    cfg = dict(cfg or {})
    if variant in ADDON_VARIANTS:
        return DLAAddon(heads, head_conv, variant, addon=cfg.get('addon'), cp=cfg.get('cp'),
                        ia=cfg.get('ia'), pretrained=cfg.get('pretrained', True))
    return MambaCenterNet(heads, head_conv, variant, neck=cfg.get('neck'), cp=cfg.get('cp'),
                          ia=cfg.get('ia'), pretrained=cfg.get('pretrained', True))


def register(model_cfg=None):
    from models import model as centernet_model
    for variant in VARIANTS + ADDON_VARIANTS:
        centernet_model._model_factory[variant] = partial(build, variant, cfg=model_cfg)
