"""CPU tests of the selective-scan reference and the scan-route utilities."""
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mambacenter.scan import (flatten_routes, gather_tokens, nonneg, priority_order,  # noqa: E402
                              restore_routes, scatter_tokens, select_tokens, selective_scan_ref,
                              soft_min, ROUTES)


def naive_scan(u, delta, A, B, C, D, delta_bias, softplus):
    b, d, length = u.shape
    groups, n = B.shape[1], A.shape[1]
    y = torch.zeros(b, d, length, dtype=torch.float64)
    for bi in range(b):
        for di in range(d):
            g = di // (d // groups)
            h = torch.zeros(n, dtype=torch.float64)
            for t in range(length):
                dt = delta[bi, di, t] + delta_bias[di]
                dt = F.softplus(dt) if softplus else dt
                h = torch.exp(dt * A[di]) * h + dt * B[bi, g, :, t] * u[bi, di, t]
                y[bi, di, t] = (h * C[bi, g, :, t]).sum() + D[di] * u[bi, di, t]
    return y


class ScanTest(unittest.TestCase):

    def test_reference_matches_naive_recurrence(self):
        torch.manual_seed(0)
        b, d, n, length, groups = 2, 6, 3, 9, 3
        u = torch.randn(b, d, length, dtype=torch.float64)
        delta = torch.randn(b, d, length, dtype=torch.float64) * 0.5
        A = -torch.rand(d, n, dtype=torch.float64) - 0.1
        B = torch.randn(b, groups, n, length, dtype=torch.float64)
        C = torch.randn(b, groups, n, length, dtype=torch.float64)
        D = torch.randn(d, dtype=torch.float64)
        bias = torch.randn(d, dtype=torch.float64) * 0.1
        for softplus in (True, False):
            dl = delta if softplus else delta.abs()
            ref = selective_scan_ref(u, dl, A, B, C, D, bias, softplus).double()
            torch.testing.assert_close(ref, naive_scan(u, dl, A, B, C, D, bias, softplus),
                                       rtol=1e-5, atol=1e-5)

    def test_routes_round_trip_and_order(self):
        x = torch.arange(6.).view(1, 1, 2, 3)
        seqs = flatten_routes(x, ROUTES)
        self.assertEqual(seqs[0, 0, 0].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(seqs[0, 1, 0].tolist(), [0, 3, 1, 4, 2, 5])
        self.assertEqual(seqs[0, 2, 0].tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(seqs[0, 3, 0].tolist(), [5, 2, 4, 1, 3, 0])
        back = restore_routes(seqs, ROUTES, 2, 3)
        for k in range(4):
            torch.testing.assert_close(back[:, k], x)

    def test_priority_order_is_bucketed_and_stable(self):
        score = torch.tensor([[0.9, 0.1, 0.5, 0.1, 0.95, 0.3]])
        # levels with 4 bins: 3, 0, 2, 0, 3, 1 -> low first, raster order inside a level
        self.assertEqual(priority_order(score, 4)[0].tolist(), [1, 3, 5, 2, 0, 4])
        self.assertEqual(priority_order(score, 4, descending=True)[0].tolist(),
                         [0, 4, 2, 5, 1, 3])
        self.assertEqual(priority_order(score, None)[0].tolist(), [1, 3, 5, 2, 0, 4])

    def test_select_tokens(self):
        torch.manual_seed(1)
        score = torch.rand(3, 200)
        idx = select_tokens(score, 0.15, 0.05, bins=4)
        self.assertEqual(idx.shape, (3, 40))
        top = score.topk(30, dim=1).indices
        for row in range(3):
            chosen = idx[row].tolist()
            self.assertEqual(len(set(chosen)), 40)
            self.assertTrue(set(top[row].tolist()) <= set(chosen))
            levels = (score[row, idx[row]] * 4).floor().clamp(0, 3)
            self.assertTrue(bool((levels[1:] >= levels[:-1]).all()))
            for level in levels.unique():
                same = idx[row][levels == level]
                self.assertTrue(bool((same[1:] > same[:-1]).all()))
        eval_idx = select_tokens(score, 0.15, 0.0)
        self.assertEqual(eval_idx.shape, (3, 30))

    def test_gather_scatter(self):
        x = torch.randn(2, 4, 10)
        idx = torch.tensor([[7, 1, 3], [0, 9, 4]])
        full = scatter_tokens(gather_tokens(x, idx), idx, 10)
        mask = torch.zeros(2, 1, 10, dtype=torch.bool)
        mask.scatter_(2, idx.unsqueeze(1), True)
        torch.testing.assert_close(full, x * mask)

    def test_nonneg_keeps_gradient(self):
        p = torch.tensor(-0.3, requires_grad=True)
        value = nonneg(p)
        value.backward()
        self.assertEqual(value.item(), 0.0)
        self.assertEqual(float(p.grad), 1.0)

    def test_soft_min_bounds(self):
        x = torch.rand(5, 4, 7) * 3
        tau = 0.5
        s = soft_min(x, tau, dim=1)
        lo = x.amin(1) - tau * math.log(4)
        self.assertTrue(bool((s <= x.amin(1) + 1e-6).all()))
        self.assertTrue(bool((s >= lo - 1e-6).all()))


if __name__ == '__main__':
    unittest.main()
