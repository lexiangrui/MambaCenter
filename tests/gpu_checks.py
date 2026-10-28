#!/usr/bin/env python
"""GPU checks before training (run through Slurm; writes a JSON report).

1. native selective_scan_cuda vs the torch reference: outputs and every gradient,
   grouped B/C, kernel softplus+bias and explicit Δ;
2. CP-SSM and IA-SSM mixers: CUDA path vs reference path (outputs and gradients);
3. CenterNet DCN on CUDA vs the float64 DCNv2 reference;
4. CenterNet VisDrone samples: stride-8 coarse targets peak at every object centre;
5. (--addon-configs) add-on detectors loaded with a trained DCN baseline checkpoint give
   the baseline's outputs on real val images, in fp32 and under bf16 autocast.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'tests'),
                str(ROOT / 'baseline/CenterNet/src/lib')]

from mambacenter import scan  # noqa: E402
from mambacenter.cp_mamba import CPConfig, CPSS2D  # noqa: E402
from mambacenter.ia_mamba import IAConfig, IASS2D  # noqa: E402


def max_rel(a, b):
    return float((a.double() - b.double()).abs().max() / (b.double().abs().max() + 1e-12))


def check_kernel(report):
    torch.manual_seed(0)
    dev = torch.device('cuda')
    b, k, d, n, length = 2, 4, 16, 16, 777
    for softplus in (True, False):
        u = torch.randn(b, k * d, length, device=dev)
        delta = torch.randn(b, k * d, length, device=dev) * 0.5
        if not softplus:
            delta = torch.nn.functional.softplus(delta)
        A = -torch.rand(k * d, n, device=dev) * 2 - 0.05
        B = torch.randn(b, k, n, length, device=dev)
        C = torch.randn(b, k, n, length, device=dev)
        D = torch.randn(k * d, device=dev)
        bias = torch.randn(k * d, device=dev) * 0.1 if softplus else None
        inputs = [t.requires_grad_() for t in (u, delta, A, B, C, D) + ((bias,) if softplus else ())]
        grad = torch.randn(b, k * d, length, device=dev)
        result = {}
        for backend in ('cuda', 'torch'):
            y = scan.selective_scan(*inputs[:6], delta_bias=bias, delta_softplus=softplus,
                                    backend=backend)
            result[backend] = (y, torch.autograd.grad(y, inputs, grad))
        errors = {'out': max_rel(result['cuda'][0], result['torch'][0])}
        names = ['u', 'delta', 'A', 'B', 'C', 'D', 'bias']
        for name, g_cuda, g_ref in zip(names, result['cuda'][1], result['torch'][1]):
            errors['d' + name] = max_rel(g_cuda, g_ref)
        report['kernel_softplus' if softplus else 'kernel_explicit_delta'] = errors
        assert max(errors.values()) < 2e-3, errors


def check_mixers(report):
    dev = torch.device('cuda')
    torch.manual_seed(0)
    cp = CPSS2D(32, CPConfig(extra_ratio=0.0), expand=1, d_state=16).to(dev).train()
    cp.set_progress(100)
    with torch.no_grad():
        cp.alpha.fill_(0.7)
    ia = IASS2D(32, 0, IAConfig(), expand=1, d_state=16).to(dev).train()
    with torch.no_grad():
        ia.gamma.fill_(0.5)
        ia.P.normal_(std=0.05)
    x = torch.randn(2, 24, 20, 32, device=dev)

    def run(module, backend, aux):
        module.backend = backend
        xx = x.clone().requires_grad_()
        y, *extras = module(xx, aux)
        loss = y.square().mean() + sum(e.mean() for e in extras)  # IA: E and L_pred
        params = [p for p in module.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, [xx] + params, allow_unused=True)
        return y, grads

    torch.manual_seed(1)
    aux = {'cp_ssm': torch.randn(2, 10, 12, 10, device=dev),      # coarse heatmap logits
           'ia_ssm': torch.rand(2, 1, 24, 20, device=dev) * 2}    # incoming innovation map
    for name, module in (('cp_ssm', cp), ('ia_ssm', ia)):
        outs = {backend: run(module, backend, aux[name]) for backend in ('cuda', 'torch')}
        errs = [max_rel(outs['cuda'][0], outs['torch'][0])]
        errs += [max_rel(a, b) for a, b in zip(outs['cuda'][1], outs['torch'][1])
                 if a is not None and b is not None]
        report[name] = {'out': errs[0], 'max_grad': max(errs[1:])}
        assert max(errs) < 5e-3, (name, errs)


def check_dcn(report):
    from test_dcn import dcn_v2_reference
    from torchvision.ops import deform_conv2d

    torch.manual_seed(0)
    cpu = [torch.randn(2, 8, 13, 11, dtype=torch.float64),
           torch.randn(2, 18, 13, 11, dtype=torch.float64) * 2,
           torch.rand(2, 9, 13, 11, dtype=torch.float64),
           torch.randn(6, 8, 3, 3, dtype=torch.float64), torch.randn(6, dtype=torch.float64)]
    cpu = [t.requires_grad_() for t in cpu]
    gpu = [t.detach().float().cuda().requires_grad_() for t in cpu]
    ref = dcn_v2_reference(*cpu)
    out = deform_conv2d(gpu[0], gpu[1], gpu[3], gpu[4], padding=1, mask=gpu[2])
    grad = torch.randn_like(ref)
    g_ref = torch.autograd.grad(ref, cpu, grad)
    g_out = torch.autograd.grad(out, gpu, grad.float().cuda())
    errors = {'out': max_rel(out.cpu(), ref)}
    errors.update({'d%d' % i: max_rel(a.cpu(), b) for i, (a, b) in enumerate(zip(g_out, g_ref))})
    report['dcn_cuda_vs_dcnv2'] = errors
    assert max(errors.values()) < 1e-4, errors


def check_targets(report, config, data_dir):
    import numpy as np
    import common

    cfg = common.load_config(config)
    cfg['loss']['coarse_weight'] = 0.5
    opt, dataset_cls = common.centernet_opt(cfg, '/tmp', data_dir)
    dataset = dataset_cls(opt, 'train')
    rng = np.random.RandomState(0)
    checked = 0
    for index in rng.choice(len(dataset), 8, replace=False):
        item = dataset[int(index)]
        hm, hm_c = item['hm'], item['hm_c']
        assert hm_c.shape == (hm.shape[0], hm.shape[1] // 2, hm.shape[2] // 2), hm_c.shape
        w = hm.shape[2]
        for ind in item['ind'][item['reg_mask'] == 1]:
            y, x = divmod(int(ind), w)
            assert hm[:, y, x].max() == 1 and hm_c[:, y // 2, x // 2].max() == 1
            checked += 1
    report['coarse_targets'] = {'objects_checked': checked, 'max_objs': dataset.max_objs}


def check_addon_parity(report, configs, baseline_ckpt, data_dir):
    import common
    import evaluate

    def heads(model, images, amp):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            out = model(images)[-1]
        return {k: out[k].float() for k in ('hm', 'wh', 'reg')}

    state = torch.load(baseline_ckpt, map_location='cpu', weights_only=False)['state_dict']
    base_cfg = common.load_config(ROOT / 'configs/centernet_dla34.yaml')
    opt, dataset_cls = common.centernet_opt(base_cfg, '/tmp', data_dir)
    base = common.create_detector(base_cfg, opt)
    base.load_state_dict(state)
    base.cuda().eval()
    loader = torch.utils.data.DataLoader(evaluate.EvalImages(dataset_cls(opt, 'val'), opt),
                                         batch_size=4, collate_fn=evaluate._collate)
    images = next(iter(loader))[0].cuda()
    ref = {amp: heads(base, images, amp) for amp in (False, True)}
    for config in configs:
        cfg = common.load_config(config)
        copt, _ = common.centernet_opt(cfg, '/tmp', data_dir)
        model = common.create_detector(cfg, copt, pretrained=False)
        result = model.load_state_dict(state, strict=False)
        new = {n for n, p in model.named_parameters()
               if any(p is q for q in model.new_parameters())}
        assert not result.unexpected_keys and set(result.missing_keys) <= new | {
            n for n, _ in model.named_buffers()}, (result.unexpected_keys, result.missing_keys)
        model.cuda().eval()
        model.set_epoch(10 ** 6)          # CP-SSM priority route and Δ modulation fully on
        errors = {}
        for amp in (False, True):
            out = heads(model, images, amp)
            for key in out:
                errors['%s_%s' % ('bf16' if amp else 'fp32', key)] = max_rel(out[key], ref[amp][key])
        report['addon_parity_' + cfg['name']] = errors
        assert max(errors.values()) < 1e-3, (config, errors)
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/centernet_dla34.yaml'))
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--addon-configs', nargs='*', default=[])
    parser.add_argument('--baseline-ckpt', help='trained DCN baseline for the add-on parity check')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'scan_import_error': scan.CUDA_IMPORT_ERROR}
    for check in (check_kernel, check_mixers, check_dcn):
        check(report)
        print(check.__name__, 'ok', flush=True)
    check_targets(report, args.config, args.data_dir)
    if args.addon_configs:
        check_addon_parity(report, args.addon_configs, args.baseline_ckpt, args.data_dir)
        print('check_addon_parity ok', flush=True)
    report['passed'] = True
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
