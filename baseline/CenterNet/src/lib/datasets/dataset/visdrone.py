from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pycocotools.coco as coco

from .coco import COCO


class VisDrone(COCO):
  """VisDrone2019-DET (COCO json 格式, 由 scripts/visdrone2coco.py 生成)

  - 10 类: 1行人 2人群 3自行车 4汽车 5面包车 6卡车 7三轮车 8三轮遮阳车 9公交 10摩托
  - 数据目录结构 (opt.data_dir/visdrone 指向 VisDrone2019-DET 根目录):
      VisDrone2019-DET-{train,val,test-dev}/{images,annotations}
      visdrone2019_det_{train,val,test-dev}_coco.json
  """
  num_classes = 10
  # VisDrone 小目标密集, 默认给 960 分辨率 (可用 --input_res 覆盖)
  default_resolution = [960, 960]
  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

  # 官方 split 目录名与 json 名的映射
  _split_dir = {'train': 'VisDrone2019-DET-train', 'val': 'VisDrone2019-DET-val',
                'test': 'VisDrone2019-DET-test-dev', 'testdev': 'VisDrone2019-DET-test-dev'}
  _split_json = {'train': 'train', 'val': 'val', 'test': 'test-dev',
                 'testdev': 'test-dev'}

  def __init__(self, opt, split):
    # 不调用 COCO.__init__ (其路径为 COCO 专有), 但保持相同属性
    self.data_dir = os.path.join(opt.data_dir, 'visdrone')
    split = 'test' if 'test' in split else split
    self.split = split
    self.opt = opt

    self.img_dir = os.path.join(self.data_dir, self._split_dir[split], 'images')
    self.annot_path = os.path.join(
        self.data_dir,
        'visdrone2019_det_{}_coco.json'.format(self._split_json[split]))
    # VisDrone 单图目标数远超 COCO (train 最多 902, val 最多 317)。上限需覆盖全部目标,
    # 否则被截断目标的中心在热图上成为负样本 (500 时 train 有 2 张图共 414 个目标被丢弃)。
    self.max_objs = 1000
    self.class_name = [
      '__background__', 'pedestrian', 'people', 'bicycle', 'car', 'van',
      'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    # COCO json 中 category_id 1..10 与输出通道 1..10 恒等映射
    self._valid_ids = list(range(1, self.num_classes + 1))
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    print('==> initializing visdrone {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))
