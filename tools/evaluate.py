#!/usr/bin/env python
"""COCO evaluation of CenterNet-family detectors with explicit maxDets.

Pre-processing, decoding and post-processing are CenterNet's: affine resize to the
input size (``fix_res``, long side = input_res) or native size (``--keep-res``),
3x3 peak NMS, top-K over classes, inverse affine to image coordinates. Scores keep
full precision. AP / AP50 / AP75 (area 'all') are reported for every maxDets value;
COCO's default protocol is maxDets=100, VisDrone's is 500.

    python tools/evaluate.py --run runs/<name> [--ckpt model_final.pth] [--split val]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import common


def preprocess(image, opt, mean, std, scale=1.0):
    """CenterNet BaseDetector.pre_process for one image -> (CHW float32, meta)."""
    from utils.image import get_affine_transform

    height, width = image.shape[:2]
    new_h, new_w = int(height * scale), int(width * scale)
    if opt.fix_res:
        inp_h, inp_w = opt.input_h, opt.input_w
        c = np.array([new_w / 2., new_h / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
    else:
        inp_h, inp_w = (new_h | opt.pad) + 1, (new_w | opt.pad) + 1
        c = np.array([new_w // 2, new_h // 2], dtype=np.float32)
        s = np.array([inp_w, inp_h], dtype=np.float32)
    trans = get_affine_transform(c, s, 0, [inp_w, inp_h])
    resized = cv2.resize(image, (new_w, new_h))
    inp = cv2.warpAffine(resized, trans, (inp_w, inp_h), flags=cv2.INTER_LINEAR)
    inp = ((inp / 255. - mean) / std).astype(np.float32).transpose(2, 0, 1)
    meta = {'c': c, 's': s, 'out_height': inp_h // opt.down_ratio,
            'out_width': inp_w // opt.down_ratio}
    return np.ascontiguousarray(inp), meta


class EvalImages(torch.utils.data.Dataset):
    def __init__(self, dataset, opt):
        self.dataset, self.opt = dataset, opt
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.dataset.images)

    def __getitem__(self, index):
        img_id = self.dataset.images[index]
        info = self.dataset.coco.loadImgs(ids=[img_id])[0]
        image = cv2.imread(os.path.join(self.dataset.img_dir, info['file_name']))
        inp, meta = preprocess(image, self.opt, self.mean, self.std)
        meta['img_id'] = img_id
        return torch.from_numpy(inp), meta


def _collate(items):
    return torch.stack([x for x, _ in items]), [m for _, m in items]


@torch.no_grad()
def predict(model, loader, opt, device):
    """{image_id: {class 1..C: (n, 5) [x1, y1, x2, y2, score]}} with at most opt.K per image."""
    from models.decode import ctdet_decode
    from utils.post_process import ctdet_post_process

    was_training = model.training
    model.eval()
    results = {}
    for images, metas in loader:
        out = model(images.to(device, non_blocking=True))[-1]
        reg = out['reg'].float() if opt.reg_offset else None
        dets = ctdet_decode(out['hm'].float().sigmoid_(), out['wh'].float(), reg=reg,
                            cat_spec_wh=opt.cat_spec_wh, K=opt.K).cpu().numpy()
        for j, meta in enumerate(metas):
            per_class = ctdet_post_process(dets[j:j + 1].copy(), [meta['c']], [meta['s']],
                                           meta['out_height'], meta['out_width'],
                                           opt.num_classes)[0]
            results[meta['img_id']] = {
                cls: np.asarray(per_class[cls], dtype=np.float32).reshape(-1, 5)
                for cls in range(1, opt.num_classes + 1)}
    model.train(was_training)
    return results


def coco_detections(results, valid_ids):
    dets = []
    for img_id, per_class in results.items():
        for cls, boxes in per_class.items():
            for x1, y1, x2, y2, score in boxes.tolist():
                dets.append({'image_id': int(img_id), 'category_id': int(valid_ids[cls - 1]),
                             'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2),
                                      round(y2 - y1, 2)],
                             'score': round(score, 6)})
    return dets


def coco_metrics(coco_gt, dets, max_dets=(100, 500)):
    """AP / AP50 / AP75 (area 'all') for each maxDets value, plus the COCO stdout summary."""
    from pycocotools.cocoeval import COCOeval

    metrics = {'n_images': len(coco_gt.getImgIds()), 'n_dets': len(dets)}
    if not dets:
        for md in max_dets:
            metrics.update({'AP@%d' % md: 0.0, 'AP50@%d' % md: 0.0, 'AP75@%d' % md: 0.0})
        return metrics, ''
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        evaluator = COCOeval(coco_gt, coco_gt.loadRes(dets), 'bbox')
        evaluator.params.maxDets = sorted({1, 10, *max_dets})
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision, ious = evaluator.eval['precision'], evaluator.params.iouThrs
    for md in max_dets:
        m = evaluator.params.maxDets.index(md)
        for name, sel in (('AP', slice(None)), ('AP50', np.isclose(ious, 0.5)),
                          ('AP75', np.isclose(ious, 0.75))):
            values = precision[sel, :, :, 0, m]
            values = values[values > -1]
            metrics['%s@%d' % (name, md)] = float(values.mean()) if values.size else float('nan')
    return metrics, log.getvalue()


class Evaluator:
    """Reusable val/test evaluation of an in-memory model (used during training)."""

    def __init__(self, opt, dataset_cls, split='val', batch_size=8, workers=4,
                 max_dets=(100, 500)):
        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset = dataset_cls(opt, split)
        self.opt, self.max_dets = opt, tuple(max_dets)
        self.loader = torch.utils.data.DataLoader(
            EvalImages(self.dataset, opt), batch_size=batch_size if opt.fix_res else 1,
            shuffle=False, num_workers=workers, pin_memory=True, collate_fn=_collate)

    def __call__(self, model, device, dets_path=None):
        start = time.time()
        dets = coco_detections(predict(model, self.loader, self.opt, device),
                               self.dataset._valid_ids)
        if dets_path:
            Path(dets_path).write_text(json.dumps(dets))
        metrics, summary = coco_metrics(self.dataset.coco, dets, self.max_dets)
        metrics['eval_seconds'] = round(time.time() - start, 1)
        return metrics, summary


def load_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = {k[7:] if k.startswith('module.') else k: v
             for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    return checkpoint.get('epoch')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='run directory with config.yaml')
    parser.add_argument('--ckpt', default='model_final.pth')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--keep-res', action='store_true', help='native-resolution test')
    parser.add_argument('--max-dets', default='100,500')
    parser.add_argument('--data-dir', default=str(common.ROOT / 'data'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tag', default='', help='suffix of the output files')
    args = parser.parse_args()

    run = Path(args.run)
    cfg = common.load_config(run / 'config.yaml')
    if args.keep_res:
        cfg['centernet_args'] = list(cfg.get('centernet_args', [])) + ['--keep_res']
    opt, dataset_cls = common.centernet_opt(cfg, run, args.data_dir)
    model = common.create_detector(cfg, opt, pretrained=False)
    epoch = load_weights(model, run / args.ckpt)
    device = torch.device('cuda')
    model.to(device)
    evaluator = Evaluator(opt, dataset_cls, args.split, args.batch_size, args.workers,
                          [int(x) for x in args.max_dets.split(',')])
    stem = 'eval_%s%s%s' % (args.split, '_native' if args.keep_res else '', args.tag)
    metrics, summary = evaluator(model, device, run / (stem + '_dets.json'))
    record = {'run': cfg['name'], 'checkpoint': args.ckpt, 'epoch': epoch, 'split': args.split,
              'test_size': 'native' if args.keep_res else 'fix_res %d' % opt.input_res,
              'decode_topk': opt.K, 'params': common.count_parameters(model), **metrics}
    (run / (stem + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(summary)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    common.setup_paths()
    sys.exit(main())
