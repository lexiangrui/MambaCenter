"""CPU tests of the add-on detectors on CenterNet's DCN neck (reference scan)."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'baseline/CenterNet/src/lib')]
from models.networks.pose_dla_dcn import DLASeg  # noqa: E402
from mambacenter.addon import DLAAddon  # noqa: E402
from mambacenter.cp_mamba import CPConfig, CPSS2D  # noqa: E402
from mambacenter.vss import SS2D  # noqa: E402
from train import param_groups  # noqa: E402

HEADS = {'hm': 10, 'wh': 2, 'reg': 2}
SMALL = dict(depth8=1, depth4=2, mlp_ratio=2.0, d_state=4, backend='torch')
VARIANTS = {'dlavss': {},
            'dlacp': {'cp': dict(warmup_epochs=1, ramp_epochs=2, sparse_ratio=0.25,
                                 alpha_init=1.0, ramp_delta=True)},
            'dlaia': {'ia': dict(cross_level=True, head_levels=(0, 1))}}


def baseline():
    torch.manual_seed(0)
    return DLASeg('dla34', HEADS, pretrained=False, down_ratio=4, final_kernel=1,
                  last_level=5, head_conv=32)


def addon(variant, base, addon_cfg=None):
    torch.manual_seed(1)
    model = DLAAddon(HEADS, 32, variant, addon={**SMALL, **(addon_cfg or {})}, pretrained=False,
                     **VARIANTS[variant])
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    new = {n for n, _ in model.named_parameters() if n.split('.')[0] in
           ('stage8', 'stage4', 'coarse', 'hm_innov')}
    assert set(result.missing_keys) >= new, set(new) - set(result.missing_keys)
    return model


def wake(model, std=0.02):
    """Give the zero-initialised projections small weights so every add-on path is live."""
    torch.manual_seed(2)
    with torch.no_grad():
        for block in list(model.stage8) + list(model.stage4):
            block.mixer.out_proj.weight.normal_(std=std)
            block.mlp[-1].weight.normal_(std=std)
        if model.hm_innov is not None:
            model.hm_innov.weight.normal_(std=std)


def total(out):
    return sum(v.float().mean() for v in out.values())


class AddonTest(unittest.TestCase):

    def test_starts_as_the_dcn_baseline(self):
        base = baseline()
        x = torch.randn(2, 3, 64, 64)
        for mode in ('train', 'eval'):
            getattr(base, mode)()
            with torch.no_grad():
                ref = base(x)[0]
            for variant in VARIANTS:
                model = getattr(addon(variant, base), mode)()
                model.set_epoch(10)
                with torch.no_grad():
                    out = model(x)[0]
                for key in HEADS:
                    torch.testing.assert_close(out[key], ref[key], rtol=1e-5, atol=1e-6,
                                               msg='%s %s %s' % (variant, mode, key))

    def test_outputs_and_gradients(self):
        base = baseline()
        x = torch.randn(2, 3, 64, 64)
        extra = {'dlavss': {}, 'dlacp': {'hm_c': (2, 10, 8, 8)}, 'dlaia': {'pred_loss': ()}}
        for variant in VARIANTS:
            model = addon(variant, base).train()
            model.set_epoch(10)
            out = model(x)[0]
            self.assertEqual(tuple(out['hm'].shape), (2, 10, 16, 16))
            for key, shape in extra[variant].items():
                self.assertEqual(tuple(out[key].shape), shape, variant + ' ' + key)
            total(out).backward()
            # Zero-initialised projections learn first ...
            for block in list(model.stage8) + list(model.stage4):
                self.assertGreater(float(block.mixer.out_proj.weight.grad.abs().sum()), 0)
            model.zero_grad()
            wake(model)
            total(model(x)[0]).backward()
            # ... then every inner parameter of the add-on receives a gradient, except the
            # modulation gains of the very first IA block, which has no earlier E to read.
            unused = ({'stage8.0.mixer.%s' % k for k in ('gamma', 'a', 'c')}
                      if variant == 'dlaia' else set())
            for name, p in model.named_parameters():
                if name.split('.')[0] in ('stage8', 'stage4', 'coarse', 'hm_innov'):
                    if name in unused:
                        self.assertIsNone(p.grad, variant + ' ' + name)
                    else:
                        self.assertIsNotNone(p.grad, variant + ' ' + name)
        cp = addon('dlacp', base)
        self.assertIsInstance(cp.stage4[0].mixer, CPSS2D)
        self.assertNotIsInstance(cp.stage8[0].mixer, CPSS2D)

    def test_ia_modulates_every_stride4_block(self):
        model = addon('dlaia', baseline()).train()
        wake(model)
        total(model(torch.randn(2, 3, 64, 64))[0]).backward()
        for block in model.stage4:
            self.assertGreater(float(block.mixer.gamma.grad.abs()), 0)
        self.assertIsNone(model.stage8[0].mixer.gamma.grad)   # first block: no earlier E

    def test_ia_without_lower_bound(self):
        base = baseline()
        x = torch.randn(2, 3, 64, 64)
        torch.manual_seed(1)
        model = DLAAddon(HEADS, 32, 'dlaia', addon=SMALL, pretrained=False,
                         ia=dict(cross_level=True, head_levels=(0, 1), lower_bound=False))
        model.load_state_dict(base.state_dict(), strict=False)
        base.eval()
        model.eval()
        with torch.no_grad():
            ref, out = base(x)[0], model(x)[0]
        for key in HEADS:
            torch.testing.assert_close(out[key], ref[key], rtol=1e-5, atol=1e-6, msg=key)
        model.train()
        wake(model)
        total(model(x)[0]).backward()
        for block in model.stage4:
            self.assertGreater(float(block.mixer.gamma.grad.abs()), 0)

    def test_new_parameters_and_lr_groups(self):
        base = baseline()
        names = {id(p): n for n, p in addon('dlaia', base).named_parameters()}
        for variant in VARIANTS:
            model = addon(variant, base)
            new = {id(p) for p in model.new_parameters()}
            expected = {id(p) for n, p in model.named_parameters()
                        if n.split('.')[0] in ('stage8', 'stage4', 'coarse', 'hm_innov')}
            self.assertEqual(new, expected, variant)
            shared = {n for n, p in model.named_parameters() if id(p) not in new}
            self.assertEqual(shared, {n for n, _ in base.named_parameters()}, variant)
            groups = param_groups(model, {'new_lr_mult': 5})
            self.assertEqual([g['lr_mult'] for g in groups], [1.0, 5.0])
            self.assertEqual(sum(len(g['params']) for g in groups),
                             len(list(model.parameters())))
            self.assertEqual(len(param_groups(model, {})), 1)
        self.assertEqual(len(param_groups(base, {'new_lr_mult': 5})), 1)
        self.assertTrue(names)

    def test_checkpoint_matches_plain_backward(self):
        base = baseline()
        x = torch.randn(2, 3, 64, 64)
        for variant in VARIANTS:
            results = []
            for flag in (False, True):
                model = addon(variant, base, {'checkpoint': flag}).train()
                wake(model)
                model.set_epoch(10)
                torch.manual_seed(5)
                out = model(x)[0]
                total(out).backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters()
                         if p.grad is not None}
                results.append((out, grads))
            (out_a, grad_a), (out_b, grad_b) = results
            for key in out_a:
                torch.testing.assert_close(out_b[key], out_a[key], msg=variant + ' ' + key)
            self.assertEqual(grad_a.keys(), grad_b.keys())
            for key in grad_a:
                torch.testing.assert_close(grad_b[key], grad_a[key], rtol=1e-4, atol=1e-6,
                                           msg=variant + ' ' + key)


    def test_micro_batch_matches_full_batch(self):
        base = baseline()
        x = torch.randn(4, 3, 64, 64)
        for variant in VARIANTS:
            results = []
            for micro in (0, 3):
                model = addon(variant, base, {'micro_batch': micro, 'checkpoint': True}).train()
                if model.cp_cfg is not None:
                    model.cp_cfg.extra_ratio = 0.0     # no random tokens: same selection
                wake(model)
                model.set_epoch(10)
                out = model(x)[0]
                total(out).backward()
                grads = {n: p.grad.clone() for n, p in model.named_parameters()
                         if p.grad is not None}
                results.append((out, grads))
            (out_a, grad_a), (out_b, grad_b) = results
            for key in out_a:
                torch.testing.assert_close(out_b[key], out_a[key], rtol=1e-5, atol=1e-6,
                                           msg=variant + ' ' + key)
            self.assertEqual(grad_a.keys(), grad_b.keys())
            for key in grad_a:
                torch.testing.assert_close(grad_b[key], grad_a[key], rtol=1e-4, atol=1e-6,
                                           msg=variant + ' ' + key)


class RampedDeltaTest(unittest.TestCase):

    def test_ramped_alpha_is_off_in_warmup(self):
        torch.manual_seed(0)
        cfg = CPConfig(regular_routes=('h', 'v'), alpha_init=1.0, ramp_delta=True,
                       warmup_epochs=2, ramp_epochs=2)
        cp = CPSS2D(8, cfg, expand=1, d_state=4, backend='torch')
        plain = SS2D(8, expand=1, d_state=4, routes=('h', 'v'), backend='torch')
        plain.load_state_dict({k: v for k, v in cp.state_dict().items()
                               if k in plain.state_dict()})
        x, hm_c = torch.randn(2, 6, 6, 8), torch.randn(2, 10, 3, 3)
        cp.set_progress(1)
        self.assertEqual(cp._gate, 0.0)
        torch.testing.assert_close(cp(x, hm_c)[0], plain(x)[0], rtol=1e-5, atol=1e-6)
        cp.set_progress(2)
        self.assertEqual(cp._gate, 0.5)
        cp.set_progress(10)
        self.assertFalse(torch.allclose(cp(x, hm_c)[0], plain(x)[0], atol=1e-4))

    def test_alpha_only_config_still_ramps(self):
        cp = CPSS2D(8, CPConfig(priority=False, alpha_init=1.0, ramp_delta=True,
                                warmup_epochs=1, ramp_epochs=1), expand=1, d_state=4,
                    backend='torch')
        cp.set_progress(0)
        self.assertEqual(cp._gate, 0.0)
        cp.set_progress(1)
        self.assertEqual(cp._gate, 1.0)


if __name__ == '__main__':
    unittest.main()
