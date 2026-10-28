#!/usr/bin/env python
"""Fold finished CenterNet-family runs into results/comparison.json.

Reads <run>/eval_val.json (written at the end of tools/train.py) and the last val
record of <run>/history.jsonl for unfinished runs; Mamba-YOLO rows are kept as they are.

    python tools/summarize.py $MC_RUNS/centernet_dla34_s317 $MC_RUNS/cpmamba_s317 ...
"""
import argparse
import datetime
import json
from pathlib import Path

import common


def run_row(run):
    run = Path(run)
    cfg = common.load_config(run / 'config.yaml')
    row = {'model': cfg['name'], 'run': run.name, 'arch': cfg['arch'], 'seed': cfg['seed'],
           'epochs_planned': cfg['train']['epochs']}
    final = run / 'eval_val.json'
    if final.exists():
        metrics = json.loads(final.read_text())
        row.update(status='completed', epoch=metrics['epoch'], params=metrics['params'])
    else:
        history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()
                   if line.strip()] if (run / 'history.jsonl').exists() else []
        evaluated = [r for r in history if 'val' in r]
        if not evaluated:
            return dict(row, status='no evaluation yet')
        metrics = evaluated[-1]['val']
        row.update(status='in progress (last val evaluation)', epoch=evaluated[-1]['epoch'])
    for md in (100, 500):
        for name in ('AP', 'AP50', 'AP75'):
            key = '%s@%d' % (name, md)
            if key in metrics:
                row[key] = metrics[key]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--index', default=str(common.ROOT / 'results' / 'comparison.json'))
    args = parser.parse_args()
    index_path = Path(args.index)
    index = json.loads(index_path.read_text())
    rows = [run_row(run) for run in args.runs]
    names = {r['run'] for r in rows}
    kept = [r for r in index.get('centernet_family', []) if r.get('run') not in names]
    index['centernet_family'] = sorted(kept + rows, key=lambda r: r['run'])
    done = {r['model'] for r in index['centernet_family'] if r['status'] == 'completed'}
    index['pending'] = [name for name in index.get('pending', []) if name not in done]
    index['updated'] = datetime.date.today().isoformat()
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + '\n')
    for r in rows:
        print('%-22s %-34s AP@100 %s AP50@100 %s AP@500 %s' % (
            r['run'], r['status'], *['%.2f' % (100 * r[k]) if k in r else '-'
                                     for k in ('AP@100', 'AP50@100', 'AP@500')]))


if __name__ == '__main__':
    main()
