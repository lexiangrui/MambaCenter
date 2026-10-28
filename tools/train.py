#!/usr/bin/env python
"""Train one CenterNet-family detector on VisDrone from an experiment config.

Every model uses CenterNet's VisDrone dataset and augmentation, CtdetLoss, decoding and
post-processing. On top of CenterNet's training loop this adds bf16 autocast, a linear
LR warm-up, an lr multiplier for parameters new to the DCN baseline (add-on models,
``train.new_lr_mult``), the epoch hook of CP-Mamba's scan schedule, periodic COCO
evaluation with explicit maxDets, a JSONL history and resumable checkpoints. A run
directory is never reused: a new experiment needs a new name, continuing one needs --resume.

    python tools/train.py --config configs/cpmamba.yaml [--set train.epochs=1 ...]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

import common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='experiment YAML (new runs)')
    parser.add_argument('--out', help='run directory (default $MC_RUNS/<name>_s<seed>)')
    parser.add_argument('--resume', action='store_true',
                        help='continue --out from its model_last.pth and config.yaml')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='config overrides for a new run')
    parser.add_argument('--data-dir', default=str(common.ROOT / 'data'),
                        help='directory containing visdrone/ (VisDrone2019-DET)')
    parser.add_argument('--max-iters', type=int, default=0,
                        help='iterations per epoch for smoke tests (0 = full epoch)')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init(_):
    import cv2
    cv2.setNumThreads(1)


def learning_rate(train, epoch, step):
    """CenterNet step schedule (x0.1 after each lr_step epoch) with a linear warm-up."""
    lr = float(train['lr']) * 0.1 ** sum(epoch > s for s in train['lr_step'])
    warmup = int(train.get('warmup_iters', 0))
    return lr * (step + 1) / warmup if step < warmup else lr


def param_groups(model, train):
    """Adam groups: everything at the schedule's lr, plus the parameters a model reports as
    new to the baseline (``new_parameters``) at ``train.new_lr_mult`` times that lr."""
    mult = float(train.get('new_lr_mult', 1))
    new = model.new_parameters() if mult != 1 and hasattr(model, 'new_parameters') else []
    ids = {id(p) for p in new}
    groups = [{'params': [p for p in model.parameters() if id(p) not in ids], 'lr_mult': 1.0}]
    if new:
        groups.append({'params': new, 'lr_mult': mult})
    return groups


def save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


class Log:
    def __init__(self, run):
        self.history = open(run / 'history.jsonl', 'a')
        self.text = open(run / 'log.txt', 'a')

    def line(self, message):
        print(message, flush=True)
        self.text.write(message + '\n')
        self.text.flush()

    def record(self, record):
        self.history.write(json.dumps(record) + '\n')
        self.history.flush()


def train_epoch(model, loader, criterion, optimizer, cfg, epoch, step, log, max_iters):
    train = cfg['train']
    device = torch.device('cuda')
    amp = train.get('amp', 'bf16') == 'bf16'
    clip = train.get('clip_grad')
    model.train()
    totals, count, skipped = defaultdict(float), 0, 0
    torch.cuda.reset_peak_memory_stats()
    start = data_end = time.time()
    data_time = 0.0
    for it, batch in enumerate(loader):
        if max_iters and it >= max_iters:
            break
        lr = learning_rate(train, epoch, step)
        for group in optimizer.param_groups:
            group['lr'] = lr * group.get('lr_mult', 1.0)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'meta'}
        data_time += time.time() - data_end
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            outputs = model(batch['input'])
        loss, stats = criterion(outputs, batch)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1
            log.line('epoch %d iter %d: non-finite loss, step skipped' % (epoch, it))
            if skipped > 20:
                raise RuntimeError('more than 20 non-finite losses in one epoch')
            data_end = time.time()
            continue
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()
        step += 1
        count += 1
        for key, value in stats.items():
            totals[key] += float(value.detach())
        if it % train.get('log_interval', 50) == 0:
            elapsed = time.time() - start
            log.line('epoch %d [%d/%d] lr %.3g | %s | %.1f img/s data %.0f%%' % (
                epoch, it, len(loader), lr,
                ' '.join('%s %.4f' % (k, totals[k] / count) for k in sorted(totals)),
                count * loader.batch_size / elapsed, 100 * data_time / elapsed))
        data_end = time.time()
    elapsed = time.time() - start
    return {**{k: v / max(count, 1) for k, v in totals.items()}, 'iters': count,
            'skipped': skipped, 'lr': lr, 'minutes': elapsed / 60,
            'img_per_s': count * loader.batch_size / elapsed,
            'max_mem_gib': torch.cuda.max_memory_allocated() / 2 ** 30}, step


def main():
    args = parse_args()
    if args.resume:
        if not args.out or args.set or args.config:
            raise SystemExit('--resume takes only --out (the saved config is reused)')
        run = Path(args.out)
        cfg = common.load_config(run / 'config.yaml')
    else:
        if not args.config:
            raise SystemExit('--config is required for a new run')
        cfg = common.load_config(args.config, args.set)
        run = Path(args.out) if args.out else common.run_dir(cfg)
        if run.exists() and any(run.iterdir()):
            raise SystemExit('%s exists: choose a new name or pass --resume' % run)
        run.mkdir(parents=True, exist_ok=True)
        (run / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    common.setup_paths()
    import evaluate
    from trains.ctdet import CtdetLoss

    log = Log(run)
    seed_everything(cfg['seed'])
    opt, dataset_cls = common.centernet_opt(cfg, run, args.data_dir)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')
    model = common.create_detector(cfg, opt).to(device)
    groups = param_groups(model, cfg['train'])
    optimizer = torch.optim.Adam(groups, lr=float(cfg['train']['lr']))
    criterion = CtdetLoss(opt)
    start_epoch, step = 0, 0
    if args.resume:
        checkpoint = torch.load(run / 'model_last.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, step = checkpoint['epoch'], checkpoint['step']
        log.line('resumed %s at epoch %d' % (run, start_epoch))
    env = {'slurm_job': os.environ.get('SLURM_JOB_ID'), 'node': os.uname().nodename,
           'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'cuda': torch.version.cuda, 'params': common.count_parameters(model),
           'lr_groups': [{'params': sum(p.numel() for p in g['params']), 'lr_mult': g['lr_mult']}
                         for g in groups],
           'resumed_at_epoch': start_epoch}
    with open(run / 'env.jsonl', 'a') as stream:
        stream.write(json.dumps(env) + '\n')
    log.line('run %s | arch %s | params %.2fM | lr groups %s | job %s on %s %s' % (
        cfg['name'], opt.arch, env['params'] / 1e6,
        ', '.join('%.2fM x%g' % (g['params'] / 1e6, g['lr_mult']) for g in env['lr_groups']),
        env['slurm_job'], env['node'], env['gpu']))

    train = cfg['train']
    workers = int(train['num_workers'])
    loader = torch.utils.data.DataLoader(
        dataset_cls(opt, 'train'), batch_size=opt.batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init,
        persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)
    ev = cfg['eval']
    evaluator = evaluate.Evaluator(opt, dataset_cls, 'val', ev['batch_size'], ev['workers'],
                                   ev['max_dets'])
    epochs = int(train['epochs'])
    for epoch in range(start_epoch + 1, epochs + 1):
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch - 1)
        stats, step = train_epoch(model, loader, criterion, optimizer, cfg, epoch, step, log,
                                  args.max_iters)
        record = {'epoch': epoch, 'step': step, 'train': stats}
        save(run / 'model_last.pth', {'epoch': epoch, 'step': step, 'state_dict': model.state_dict(),
                                      'optimizer': optimizer.state_dict()})
        if epoch in train['lr_step']:
            save(run / ('model_%d.pth' % epoch), {'epoch': epoch, 'state_dict': model.state_dict()})
        final = epoch == epochs
        if final or epoch % int(ev['interval']) == 0:
            metrics, summary = evaluator(model, device,
                                         run / 'eval_val_dets.json' if final else None)
            record['val'] = metrics
            log.line(summary.strip())
        log.record(record)
        log.line('epoch %d done: %s' % (epoch, json.dumps(
            {k: float('%.4g' % v) if isinstance(v, float) else v
             for k, v in {**stats, **record.get('val', {})}.items()})))
        if final:
            save(run / 'model_final.pth', {'epoch': epochs, 'state_dict': model.state_dict()})
            result = {'run': cfg['name'], 'arch': opt.arch, 'epoch': epochs, 'split': 'val',
                      'test_size': 'fix_res %d' % opt.input_res, 'decode_topk': opt.K,
                      'params': common.count_parameters(model), **metrics}
            (run / 'eval_val.json').write_text(json.dumps(result, indent=2) + '\n')
            log.line('final: ' + json.dumps(result))


if __name__ == '__main__':
    sys.exit(main())
