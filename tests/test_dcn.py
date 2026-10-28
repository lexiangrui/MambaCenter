"""The baseline's torchvision DCN equals the original CenterNet DCNv2 (modulated deformable conv).

The reference re-implements DCNv2's CUDA im2col (CharlesShang/DCNv2, bundled with
CenterNet): offset channels (2k, 2k+1) = (Δh, Δw) of kernel tap k per deformable group,
mask channel k, bilinear sampling with zero padding outside the image. Forward values
and all gradients are compared in float64.
"""
import sys
import unittest
from pathlib import Path

import torch
from torchvision.ops import deform_conv2d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baseline/CenterNet/src/lib'))
from models.networks.DCNv2.dcn_v2 import DCN  # noqa: E402


def bilinear(img, h, w):
    """DCNv2 dmcn_im2col_bilinear: img (b, c, H, W), h/w (b, Ho, Wo) -> (b, c, Ho, Wo)."""
    b, c, H, W = img.shape
    inside = (h > -1) & (w > -1) & (h < H) & (w < W)
    h0, w0 = torch.floor(h), torch.floor(w)
    lh, lw = h - h0, w - w0
    out = 0
    for dh, dw, weight in ((0, 0, (1 - lh) * (1 - lw)), (0, 1, (1 - lh) * lw),
                           (1, 0, lh * (1 - lw)), (1, 1, lh * lw)):
        hi, wi = (h0 + dh).long(), (w0 + dw).long()
        valid = inside & (hi >= 0) & (hi <= H - 1) & (wi >= 0) & (wi <= W - 1)
        flat = (hi.clamp(0, H - 1) * W + wi.clamp(0, W - 1)).flatten(1)
        v = img.flatten(2).gather(2, flat.unsqueeze(1).expand(-1, c, -1)).view(b, c, *h.shape[1:])
        out = out + v * (weight * valid).unsqueeze(1)
    return out


def dcn_v2_reference(x, offset, mask, weight, bias, stride=1, padding=1, dilation=1, groups=1):
    b, c, H, W = x.shape
    cout, _, kh, kw = weight.shape
    Ho = (H + 2 * padding - (dilation * (kh - 1) + 1)) // stride + 1
    Wo = (W + 2 * padding - (dilation * (kw - 1) + 1)) // stride + 1
    base_h = (torch.arange(Ho, dtype=x.dtype) * stride - padding).view(1, Ho, 1)
    base_w = (torch.arange(Wo, dtype=x.dtype) * stride - padding).view(1, 1, Wo)
    cg = c // groups
    cols = []
    for i in range(kh):
        for j in range(kw):
            k = i * kw + j
            taps = []
            for g in range(groups):
                dh = offset[:, g * 2 * kh * kw + 2 * k]
                dw = offset[:, g * 2 * kh * kw + 2 * k + 1]
                m = mask[:, g * kh * kw + k].unsqueeze(1)
                taps.append(bilinear(x[:, g * cg:(g + 1) * cg], base_h + i * dilation + dh,
                                     base_w + j * dilation + dw) * m)
            cols.append(torch.cat(taps, 1))
    cols = torch.stack(cols, 2)
    return torch.einsum('ock,bckhw->bohw', weight.flatten(2), cols) + bias.view(1, -1, 1, 1)


class DCNTest(unittest.TestCase):

    def test_torchvision_op_equals_dcnv2(self):
        torch.manual_seed(0)
        for groups in (1, 2):
            x = torch.randn(2, 4, 7, 9, dtype=torch.float64, requires_grad=True)
            offset = (torch.randn(2, 2 * groups * 9, 7, 9, dtype=torch.float64) * 2).requires_grad_()
            mask = torch.rand(2, groups * 9, 7, 9, dtype=torch.float64, requires_grad=True)
            weight = torch.randn(5, 4, 3, 3, dtype=torch.float64, requires_grad=True)
            bias = torch.randn(5, dtype=torch.float64, requires_grad=True)
            inputs = (x, offset, mask, weight, bias)
            ref = dcn_v2_reference(*inputs, groups=groups)
            out = deform_conv2d(x, offset, weight, bias, padding=1, mask=mask)
            torch.testing.assert_close(out, ref)
            grad = torch.randn_like(ref)
            g_ref = torch.autograd.grad(ref, inputs, grad)
            g_out = torch.autograd.grad(out, inputs, grad)
            for a, r in zip(g_out, g_ref):
                torch.testing.assert_close(a, r)

    def test_centernet_dcn_module_semantics(self):
        """DCN module: offset = cat(o1, o2), mask = sigmoid(m) from conv_offset_mask."""
        torch.manual_seed(1)
        module = DCN(4, 6, kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                     deformable_groups=1).double()
        torch.nn.init.normal_(module.conv_offset_mask.weight, std=0.3)
        x = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        o1, o2, m = torch.chunk(module.conv_offset_mask(x), 3, dim=1)
        ref = dcn_v2_reference(x, torch.cat((o1, o2), 1), torch.sigmoid(m), module.weight,
                               module.bias)
        torch.testing.assert_close(module(x), ref)
        self.assertEqual(float(DCN(4, 6, (3, 3), 1, 1).conv_offset_mask.weight.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
