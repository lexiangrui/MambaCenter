#!/usr/bin/env python
"""Peak GPU memory and training-step time of experiment configs at several batch sizes.

Each config builds its model, CtdetLoss and Adam exactly as tools/train.py does and
runs full training steps (forward, backward, optimizer) under the config's precision
on real augmented VisDrone samples. CP-Mamba runs with its priority route enabled
(worst case). Batch sizes are tried in ascending order until the first OOM.

    python tools/probe.py --configs configs/a.yaml configs/b.yaml --batch-sizes 8,16,24
"""
import argparse
import contextlib
import io
import json
import statistics
import sys
import time

import torch

import common


def probe(config, batch_sizes, steps, warmup, data_dir):
    from trains.ctdet import CtdetLoss

    cfg = common.load_config(config)
    opt, dataset_cls = common.centernet_opt(cfg, '/tmp', data_dir)
    torch.manual_seed(0)
    model = common.create_detector(cfg, opt).cuda().train()
    if hasattr(model, 'set_epoch'):
        model.set_epoch(10 ** 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = CtdetLoss(opt)
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = dataset_cls(opt, 'train')
    samples = [dataset[i] for i in range(max(batch_sizes))]
    amp = cfg['train'].get('amp', 'bf16') == 'bf16'
    rows = []
    for size in sorted(batch_sizes):
        batch = {k: torch.stack([torch.as_tensor(s[k]) for s in samples[:size]]).cuda()
                 for k in samples[0] if k != 'meta'}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            for step in range(warmup + steps):
                torch.cuda.synchronize()
                start = time.time()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    outputs = model(batch['input'])
                loss, _ = criterion(outputs, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step >= warmup:
                    times.append(time.time() - start)
            step_s = statistics.median(times)
            rows.append({'batch_size': size, 'step_s': round(step_s, 4),
                         'img_per_s': round(size / step_s, 1),
                         'peak_alloc_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                         'peak_reserved_gib': round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
                         'loss': round(float(loss), 4)})
        except torch.cuda.OutOfMemoryError:
            rows.append({'batch_size': size, 'oom': True})
            break
        finally:
            del batch
            outputs = loss = None
        print(config, rows[-1], flush=True)
    return {'config': config, 'arch': opt.arch, 'params': common.count_parameters(model),
            'input_res': opt.input_res, 'amp': amp, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--batch-sizes', default='8,12,16,20,24,28,32')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--data-dir', default=str(common.ROOT / 'data'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    common.setup_paths()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    sizes = [int(x) for x in args.batch_sizes.split(',')]
    report = {'gpu': torch.cuda.get_device_name(),
              'total_gib': round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2),
              'results': []}
    for config in args.configs:
        report['results'].append(probe(config, sizes, args.steps, args.warmup, args.data_dir))
        torch.cuda.empty_cache()
        with open(args.out, 'w') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
