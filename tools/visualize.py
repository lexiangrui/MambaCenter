#!/usr/bin/env python
"""Draw a run's detections on a few val images (boxes, class, score).

    python tools/visualize.py --run $MC_RUNS/<run> [--ckpt model_final.pth] [--n 6] [--thresh 0.3]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import common
import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--ckpt', default='model_final.pth')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--thresh', type=float, default=0.3)
    parser.add_argument('--out', help='output directory (default <run>/vis)')
    parser.add_argument('--data-dir', default=str(common.ROOT / 'data'))
    args = parser.parse_args()

    run = Path(args.run)
    cfg = common.load_config(run / 'config.yaml')
    opt, dataset_cls = common.centernet_opt(cfg, run, args.data_dir)
    model = common.create_detector(cfg, opt, pretrained=False)
    evaluate.load_weights(model, run / args.ckpt)
    device = torch.device('cuda')
    model.to(device)
    dataset = dataset_cls(opt, 'val')
    images = evaluate.EvalImages(dataset, opt)
    pick = np.linspace(0, len(images) - 1, args.n).astype(int)
    subset = torch.utils.data.Subset(images, pick.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=1, collate_fn=evaluate._collate)
    results = evaluate.predict(model, loader, opt, device)
    out = Path(args.out or run / 'vis')
    out.mkdir(parents=True, exist_ok=True)
    names = dataset.class_name[1:]
    for img_id, per_class in results.items():
        info = dataset.coco.loadImgs(ids=[img_id])[0]
        image = cv2.imread(os.path.join(dataset.img_dir, info['file_name']))
        for cls, boxes in per_class.items():
            color = tuple(int(c) for c in dataset.voc_color[cls - 1])
            for x1, y1, x2, y2, score in boxes[boxes[:, 4] >= args.thresh]:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, '%s %.2f' % (names[cls - 1], score), (int(x1), int(y1) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(out / Path(info['file_name']).name), image)
    print('wrote %d images to %s' % (len(results), out))


if __name__ == '__main__':
    common.setup_paths()
    sys.exit(main())
