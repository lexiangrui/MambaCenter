"""CPU tests of the Mamba-FPN / CP-Mamba / IA-Mamba mechanisms with the reference scan."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'baseline/CenterNet/src/lib')]
from mambacenter.cp_mamba import CPConfig, CPSS2D  # noqa: E402
from mambacenter.detector import MambaCenterNet  # noqa: E402
from mambacenter.ia_mamba import IAConfig, IASS2D  # noqa: E402
from mambacenter.scan import ROUTES  # noqa: E402
from mambacenter.vss import SS2D  # noqa: E402

HEADS = {'hm': 10, 'wh': 2, 'reg': 2}
TINY = dict(dims=(16, 16, 16, 16), depths=(1, 2, 1, 1), expands=(1, 1, 1, 1), d_state=4,
            backend='torch')


def tiny(variant, neck=None, **kw):
    torch.manual_seed(0)
    return MambaCenterNet(HEADS, 32, variant, neck={**TINY, **(neck or {})}, pretrained=False, **kw)


def copy_common(src, dst):
    state = {k: v for k, v in src.state_dict().items() if k in dst.state_dict()}
    missing = dst.load_state_dict(state, strict=False).missing_keys
    if missing:
        raise AssertionError(missing)


class DetectorTest(unittest.TestCase):

    def check_outputs(self, model, extra):
        x = torch.randn(2, 3, 64, 64)
        out = model(x)[0]
        self.assertEqual(tuple(out['hm'].shape), (2, 10, 16, 16))
        self.assertEqual(tuple(out['wh'].shape), (2, 2, 16, 16))
        self.assertEqual(tuple(out['reg'].shape), (2, 2, 16, 16))
        for key, shape in extra.items():
            self.assertEqual(tuple(out[key].shape), shape)
        loss = sum(v.float().mean() for v in out.values())
        loss.backward()
        return out

    def test_mambafpn(self):
        self.check_outputs(tiny('mambafpn').train(), {})

    def test_cpmamba_gradients_and_schedule(self):
        model = tiny('cpmamba', cp=dict(warmup_epochs=1, ramp_epochs=2, sparse_ratio=0.25))
        block = model.neck.levels[0][0].mixer
        model.set_epoch(0)
        self.assertEqual(block._gate, 0.0)
        model.set_epoch(1)
        self.assertEqual(block._gate, 0.5)
        model.set_epoch(5)
        self.assertEqual(block._gate, 1.0)
        model.train()
        self.check_outputs(model, {'hm_c': (2, 10, 8, 8)})
        self.assertIsNotNone(block.alpha.grad)
        self.assertIsNotNone(block.prio.x_proj.grad)
        self.assertGreater(float(model.coarse.head[0].weight.grad.abs().sum()), 0)
        model.set_epoch(1)
        fresh = tiny('cpmamba', cp=dict(warmup_epochs=1, ramp_epochs=2, sparse_ratio=0.25))
        fresh.load_state_dict(model.state_dict())
        self.assertEqual(fresh.neck.levels[0][0].mixer._gate, 0.5)

    def test_cpmamba_b1_uses_plain_vss(self):
        model = tiny('cpmamba', cp=dict(use_cp_block=False)).train()
        self.assertNotIsInstance(model.neck.levels[0][0].mixer, CPSS2D)
        self.check_outputs(model, {'hm_c': (2, 10, 8, 8)})

    def test_iamamba(self):
        model = tiny('iamamba').train()
        out = self.check_outputs(model, {'pred_loss': ()})
        self.assertGreaterEqual(float(out['pred_loss']), 0.0)
        block = model.neck.levels[1][1].mixer   # second P3 block receives E of the first
        self.assertIsNotNone(block.gamma.grad)
        self.assertIsNotNone(block.P.grad)
        self.assertEqual(model.hm[0].in_channels, 16 + 3)
        model.eval()
        with torch.no_grad():
            self.assertNotIn('pred_loss', model(torch.randn(1, 3, 64, 64))[0])


class MixerEquivalenceTest(unittest.TestCase):
    """Zero-initialised gains make the new mixers start as the plain SS2D."""

    def test_ia_with_zero_gamma_equals_ss2d(self):
        torch.manual_seed(0)
        ia = IASS2D(8, level=0, cfg=IAConfig(), expand=1, d_state=4,
                    backend='torch')
        plain = SS2D(8, expand=1, d_state=4, routes=ROUTES, backend='torch')
        copy_common(ia, plain)
        x = torch.randn(2, 5, 6, 8)
        y, E, pred = ia(x, torch.rand(2, 1, 5, 6) * 3)
        torch.testing.assert_close(y, plain(x)[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(tuple(E.shape), (2, 1, 5, 6))
        self.assertEqual(pred.dim(), 0)

    def test_cp_with_zero_alpha_and_gate_equals_ss2d(self):
        torch.manual_seed(0)
        cp = CPSS2D(8, CPConfig(regular_routes=ROUTES), expand=1, d_state=4, backend='torch')
        cp.set_progress(0)
        plain = SS2D(8, expand=1, d_state=4, routes=ROUTES, backend='torch')
        copy_common(cp, plain)
        x = torch.randn(2, 6, 6, 8)
        torch.testing.assert_close(cp(x, torch.randn(2, 10, 3, 3))[0], plain(x)[0],
                                   rtol=1e-5, atol=1e-6)


class CheckpointTest(unittest.TestCase):
    """Recomputing blocks in backward gives the same outputs and gradients."""

    def test_checkpoint_matches_plain_backward(self):
        cases = (('mambafpn', {}), ('cpmamba', {'cp': {'warmup_epochs': 0, 'sparse_ratio': 0.25}}),
                 ('iamamba', {}))
        x = torch.randn(2, 3, 64, 64)
        for variant, kw in cases:
            results = []
            for flag in (False, True):
                model = tiny(variant, neck={'checkpoint': flag}, **kw).train()
                if results:
                    model.load_state_dict(results[0][2])
                model.set_epoch(10)
                torch.manual_seed(5)
                out = model(x)[0]
                sum(v.float().mean() for v in out.values()).backward()
                grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
                results.append((out, grads, model.state_dict()))
            (out_a, grad_a, _), (out_b, grad_b, _) = results
            for key in out_a:
                torch.testing.assert_close(out_b[key], out_a[key], msg=variant + ' ' + key)
            self.assertEqual(grad_a.keys(), grad_b.keys())
            for key in grad_a:
                torch.testing.assert_close(grad_b[key], grad_a[key], rtol=1e-4, atol=1e-6,
                                           msg=variant + ' ' + key)


if __name__ == '__main__':
    unittest.main()
