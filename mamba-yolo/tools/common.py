import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/Mamba-YOLO'))
import torch
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import ultralytics.data.utils as data_utils
data_utils.check_font = lambda *args, **kwargs: None  # plots are disabled; no compute-node font download
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import de_parallel

class MatchedValidator(DetectionValidator):
    """Square FP32 validation with the shared VisDrone COCO annotations/evaluator."""
    def build_dataset(self, img_path, mode='val', batch=None):
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=self.stride)

    def init_metrics(self, model):
        super().init_metrics(model)
        from pycocotools.coco import COCO
        self.gt = COCO(self.data['coco_gt'])
        self.image_ids = {Path(im['file_name']).name:im['id'] for im in self.gt.dataset['images']}
        self.class_map = [c['id'] for c in sorted(self.gt.dataset['categories'], key=lambda c:c['id'])]
        assert len(self.class_map) == 10
        self.args.save_json = True

    def pred_to_json(self, predn, filename):
        image_id = self.image_ids[Path(filename).name]
        box = ops.xyxy2xywh(predn[:, :4]); box[:, :2] -= box[:, 2:] / 2
        for p,b in zip(predn.tolist(), box.tolist()):
            if b[2] > 0 and b[3] > 0:
                self.jdict.append({'image_id':image_id,'category_id':self.class_map[int(p[5])],
                                  'bbox':b,'score':p[4]})

    def eval_json(self, stats):
        from pycocotools.cocoeval import COCOeval
        from pycocotools.coco import COCO
        if self.jdict:
            dt = self.gt.loadRes(self.jdict)
        else:
            dt = COCO(); dt.dataset = {**self.gt.dataset,'annotations':[]}; dt.createIndex()
        evaluator = COCOeval(self.gt,dt,'bbox')
        evaluator.params.imgIds = [self.image_ids[Path(x).name] for x in self.dataloader.dataset.im_files]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        self.coco_metrics = {k:float(evaluator.stats[i]) for k,i in [('AP',0),('AP50',1),('AP75',2)]}
        self.coco_metrics.update(n_images=len(evaluator.params.imgIds),n_dets=len(self.jdict),
                                 coco_max_dets=evaluator.params.maxDets,decode_topk=self.args.max_det)
        stats[self.metrics.keys[2]] = self.coco_metrics['AP50']
        stats[self.metrics.keys[3]] = self.coco_metrics['AP']
        return stats

class MatchedTrainer(DetectionTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode=='val':batch_size=min(batch_size,2)
        return super().get_dataloader(dataset_path,batch_size,rank,mode)

    def final_eval(self):
        # Native final_eval reloads an FP16-rounded best checkpoint and repeats callbacks.
        # Dedicated evaluation uses our saved FP32 epoch70/final checkpoints instead.
        pass

    def build_dataset(self, img_path, mode='train', batch=None):
        gs=max(int(de_parallel(self.model).stride.max()),32)
        return build_yolo_dataset(self.args,img_path,batch,self.data,mode=mode,rect=False,stride=gs)

    def get_validator(self):
        from copy import copy
        self.loss_names=('box_loss','cls_loss','dfl_loss')
        return MatchedValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)

    def validate(self):
        metrics=self.validator(self)
        metrics=self.validator.eval_json(metrics)
        metrics.pop('fitness',None)
        fitness=self.validator.coco_metrics['AP']
        if not self.best_fitness or self.best_fitness < fitness:self.best_fitness=fitness
        return metrics,fitness
