#!/usr/bin/env python
"""VisDrone2019-DET 原始标注 -> COCO json

VisDrone 标注格式 (每行): bbox_left,bbox_top,bbox_w,bbox_h,score,category,truncation,occlusion
类别: 0=忽略 1=行人 2=人群 3=自行车 4=汽车 5=面包车 6=卡车 7=三轮车 8=三轮遮阳车 9=公交 10=摩托 11=忽略区域
转换规则:
  - 保留类别 1-10, 映射为 COCO category_id 1-10; 跳过 0/11
  - 跳过 w/h<=0 的框; 框裁剪到图像边界内
  - iscrowd 全部为 0
用法: python scripts/visdrone2coco.py --root data/VisDrone2019-DET --splits train val test-dev
"""
import argparse
import json
import os

from PIL import Image

CATEGORIES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}

SPLIT_DIRS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val",
              "test-dev": "VisDrone2019-DET-test-dev"}


def convert_split(root: str, split: str) -> dict:
    split_dir = os.path.join(root, SPLIT_DIRS[split])
    img_dir = os.path.join(split_dir, "images")
    ann_dir = os.path.join(split_dir, "annotations")
    out_json = os.path.join(root, f"visdrone2019_det_{split}_coco.json")

    images, annotations = [], []
    ann_id = 1
    skipped = 0
    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    for img_id, fname in enumerate(img_files, start=1):
        with Image.open(os.path.join(img_dir, fname)) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        ann_file = os.path.join(ann_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(ann_file):
            continue
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                x, y, bw, bh, _score, cat = (int(parts[i]) for i in range(6))
                if cat not in CATEGORIES:          # 0/11 忽略
                    skipped += 1
                    continue
                x2, y2 = min(x + bw, w), min(y + bh, h)
                x, y = max(0, x), max(0, y)
                bw, bh = x2 - x, y2 - y
                if bw <= 0 or bh <= 0:
                    skipped += 1
                    continue
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cat,
                    "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
                })
                ann_id += 1

    coco = {
        "info": {"description": f"VisDrone2019-DET {split} in COCO format"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name, "supercategory": "object"}
                       for cid, name in CATEGORIES.items()],
    }
    with open(out_json, "w") as f:
        json.dump(coco, f)
    n_boxes_by_cat = {}
    for a in annotations:
        n_boxes_by_cat[a["category_id"]] = n_boxes_by_cat.get(a["category_id"], 0) + 1
    print(f"[{split}] {len(images)} imgs, {len(annotations)} boxes "
          f"(跳过 {skipped} 个忽略/无效框) -> {out_json}")
    for cid in sorted(n_boxes_by_cat):
        print(f"    cat {cid} {CATEGORIES[cid]:16s}: {n_boxes_by_cat[cid]}")
    return coco


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="VisDrone2019-DET dataset root")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test-dev"])
    args = ap.parse_args()
    for s in args.splits:
        convert_split(args.root, s)
