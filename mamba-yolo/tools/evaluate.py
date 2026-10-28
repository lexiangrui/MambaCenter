import argparse,json,statistics
from pathlib import Path
from common import MatchedValidator
import torch
from ultralytics.utils import ops

p=argparse.ArgumentParser();p.add_argument('--ckpt',required=True);p.add_argument('--data',required=True)
p.add_argument('--out',required=True);p.add_argument('--reference-summary',default='');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
ckpt=torch.load(a.ckpt,map_location='cpu',weights_only=False)
model=ckpt['model'].float().cuda().eval()
params=sum(p.numel() for p in model.parameters())
v=MatchedValidator(save_dir=out,args={'data':a.data,'imgsz':960,'batch':2,'workers':8,'device':'0',
    'conf':.01,'iou':.7,'max_det':500,'half':False,'plots':False,'save_json':True,'rect':False})
metrics=v(model=model)
if not getattr(v,'coco_metrics',None):v.eval_json(metrics)
record={'checkpoint':a.ckpt,'epoch':ckpt['epoch'],'params':params,'metrics':v.coco_metrics,
        'protocol':'VisDrone COCO-style bbox evaluation; maxDets100/category; max500/image; FP32 square960; native preprocessing',
        'source_initialization':'Mamba-YOLO official scratch',
        'scope':'cross-architecture experiment; different optimizer, augmentations and pretraining'}
if a.reference_summary:
    reference=json.loads(Path(a.reference_summary).read_text())
    record['AP_differences_pp']={f"{r['model']}@{r['epoch']}":
        (record['metrics']['AP']-r['AP'])*100 for r in reference['models']
        if r['model'] != 'Mamba-YOLO-B' and r.get('status') != 'retired-code'}
(out/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
# Fresh, unfused FP32 model: include official decode and NMS in timing.
model=torch.load(a.ckpt,map_location='cpu',weights_only=False)['model'].float().cuda().eval()
x=torch.rand(1,3,960,960,device='cuda')
def infer():return ops.non_max_suppression(model(x),.01,.7,max_det=500,multi_label=True)
with torch.inference_mode():
    for _ in range(10):infer()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(50):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();infer();end.record();end.synchronize();times.append(start.elapsed_time(end))
benchmark={'params':params,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
    'input_size':960,'batch_size':1,'precision':'FP32','fused':False,
    'scope':'forward + native decode + NMS; excludes preprocessing/I/O',
    'median_ms':statistics.median(times),'mean_ms':statistics.mean(times),'p90_ms':sorted(times)[44],
    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
(out/'benchmark.json').write_text(json.dumps(benchmark,indent=2)+'\n')
print(json.dumps(record,indent=2));print(json.dumps(benchmark,indent=2))
