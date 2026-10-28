import argparse,json,shutil
from copy import deepcopy
from pathlib import Path
from common import ROOT,MatchedTrainer
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
import ultralytics.engine.trainer as trainer_module

p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
p.add_argument('--epochs',type=int,default=300);p.add_argument('--batch',type=int,default=2)
p.add_argument('--size',type=int,default=960);p.add_argument('--workers',type=int,default=8)
p.add_argument('--resume', default='', help='Explicit trusted last.pt checkpoint; no automatic resume')
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'train').exists() and not a.resume:raise FileExistsError(f'Use a fresh run directory: {out}')
if a.resume and not Path(a.resume).is_file():raise FileNotFoundError(a.resume)
init_seeds(317,deterministic=True)
# Real-model FP32/AMP and loss/gradient smoke checks gate submission; do not download unrelated YOLO AMP-check weights.
trainer_module.check_amp=lambda model:True
model=YOLO(a.resume or str(ROOT/'configs/Mamba-YOLO-B-visdrone.yaml'))
params=sum(v.numel() for v in model.model.parameters())
assert params==21800910,params
(out/('resume_provenance.json' if a.resume else 'model_provenance.json')).write_text(json.dumps({'model':'official Mamba-YOLO-B, default width0.50','nc':10,'params':params,
    'official_B_width':0.50,'width':0.50,'official_B_visdrone_params':21800910,
    'source_commit':'51f7be9bd5ab907b6f7d2c91bb71ca5a4be063aa','initialization':'official from scratch',
    'warning':'Equal-epoch comparison is not identical pretraining or optimization.',
    'scan':'official selective_scan_cuda_core, sm120','torch':str(torch.__version__)},indent=2)+'\n')

def on_epoch_end(t):
    record={'epoch':t.epoch,'metrics':t.metrics,'loss':[float(x) for x in t.tloss],
            'coco':getattr(t.validator,'coco_metrics',None)}
    with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if t.epoch==69 or t.epoch==t.epochs-1:
        target=t.wdir/('epoch70_fp32.pt' if t.epoch==69 else 'final_fp32.pt')
        torch.save({'model':deepcopy(t.ema.ema).float().cpu(),'epoch':t.epoch,
                    'train_args':vars(t.args)},target)
    if t.loss is not None and not torch.isfinite(t.loss):raise RuntimeError('Nonfinite training loss')
model.add_callback('on_fit_epoch_end',on_epoch_end)
def on_epoch_start(t):
    t.args.val=(t.epoch+1)%5==0 or t.epoch==t.epochs-1
    t.validator.coco_metrics=None
model.add_callback('on_train_epoch_start',on_epoch_start)
def on_batch_end(t):
    if not torch.isfinite(t.loss).all():raise RuntimeError(f'Nonfinite loss at epoch {t.epoch}')
model.add_callback('on_train_batch_end',on_batch_end)
def verify_resume_directory(t):
    if a.resume and t.save_dir.resolve() != (out/'train').resolve():
        raise ValueError(f'Resume output differs from checkpoint output: {t.save_dir}')
model.add_callback('on_pretrain_routine_end', verify_resume_directory)
model.train(trainer=MatchedTrainer,resume=a.resume or False,data=a.data,epochs=a.epochs,imgsz=a.size,batch=a.batch,
    nbs=16,optimizer='SGD',lr0=0.01,lrf=0.01,momentum=0.937,weight_decay=0.0005,
    warmup_epochs=3.0,seed=317,deterministic=True,pretrained=False,device=0,workers=a.workers,
    amp=True,cache=False,rect=False,multi_scale=False,max_det=500,conf=0.01,iou=0.7,
    half=False,plots=False,val=True,save=True,save_period=10,patience=0,
    project=str(out),name='train',exist_ok=False)
