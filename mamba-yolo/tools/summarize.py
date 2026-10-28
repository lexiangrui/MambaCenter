"""Summarize recorded full-val AP/AP50 without restarting training or evaluating weights."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', required=True, help='Existing experiment folder containing history.jsonl')
p.add_argument('--reference-summary', default='')
a = p.parse_args()
out = Path(a.out)
history = [json.loads(line) for line in (out/'history.jsonl').read_text().splitlines() if line.strip()]
evaluated = [r for r in history if r.get('coco') and r['coco']['n_images'] == 548]
selected = {r['epoch']: r for r in evaluated if r['epoch'] == 69 or r is evaluated[-1]}
rows = []
if a.reference_summary:
    reference = json.loads(Path(a.reference_summary).read_text())
    rows += [r for r in reference['models'] if r['model'] != 'Mamba-YOLO-B' and r.get('status') != 'retired-code']
for epoch, r in sorted(selected.items()):
    rows.append({'model': 'Mamba-YOLO-B', 'epoch': epoch+1, 'AP': r['coco']['AP'], 'AP50': r['coco']['AP50']})
completed = max((r['epoch']+1 for r in history), default=0)
report = {'epochs_completed': completed, 'budget_completed': completed == 300, 'rows': rows,
          'protocol': 'VisDrone548, input960, overall COCO-style AP/AP50; maxDets100/category.',
          'limitations': ['Epoch70 uses a 300-epoch learning-rate schedule.',
                         'Models differ in initialization, augmentation, optimizer, losses and EMA.',
                         'Incomplete budget does not by itself establish whether a job is running or cancelled.']}
(out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
table = ['| Model | Epoch | AP (%) | AP50 (%) |', '|---|---:|---:|---:|']
table += [f"| {r['model']} | {r['epoch']} | {100*r['AP']:.2f} | {100*r['AP50']:.2f} |" for r in rows]
(out/'comparison.md').write_text('\n'.join(table)+'\n')
print(json.dumps(report, indent=2))
