"""Create YOLO labels from VisDrone COCO annotations; reuse images via symlinks."""
import argparse,json,collections
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--source',required=True,help='VisDrone2019-DET dataset root');p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=0)
a=p.parse_args();out=Path(a.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
source=Path(a.source).expanduser().resolve();counts={}
for split in ['train','val']:
    ann=source/f'visdrone2019_det_{split}_coco.json';ds=json.loads(ann.read_text())
    images=ds['images'][:a.limit] if a.limit else ds['images']
    grouped=collections.defaultdict(list)
    for box in ds['annotations']:grouped[box['image_id']].append(box)
    cats=sorted(ds['categories'],key=lambda c:c['id']);ids=[c['id'] for c in cats]
    assert ids==list(range(1,11))
    (out/'images'/split).mkdir(parents=True,exist_ok=True);(out/'labels'/split).mkdir(parents=True,exist_ok=True)
    n=0
    for im in images:
        name=Path(im['file_name']).name
        original=source/f'VisDrone2019-DET-{split}'/'images'/name
        assert original.is_file(),original
        dest=out/'images'/split/name
        if not dest.exists():dest.symlink_to(original)
        lines=[]
        for box in grouped[im['id']]:
            x,y,w,h=box['bbox'];assert w>0 and h>0
            lines.append(f"{ids.index(box['category_id'])} {(x+w/2)/im['width']:.10f} {(y+h/2)/im['height']:.10f} {w/im['width']:.10f} {h/im['height']:.10f}")
        (out/'labels'/split/Path(name).with_suffix('.txt').name).write_text('\n'.join(lines)+'\n')
        n+=len(lines)
    counts[split]={'images':len(images),'objects':n}
cfg={'path':str(out),'train':'images/train','val':'images/val','names':[c['name'] for c in cats],
     'coco_gt':str(source/'visdrone2019_det_val_coco.json')}
(out/'dataset.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(out/'manifest.json').write_text(json.dumps(counts,indent=2)+'\n')
print(counts)
