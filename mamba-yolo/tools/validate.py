import argparse,json,time
from pathlib import Path
from common import ROOT
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.common_utils_mbyolo import SelectiveScanCore
from ultralytics.utils.torch_utils import init_seeds

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
init_seeds(317,deterministic=True);torch.set_num_threads(4)
torch.cuda.set_device(0)
base=[torch.randn(2,8,23,device='cuda')*.1,torch.randn(2,8,23,device='cuda')*.1,
      -torch.rand(8,3,device='cuda'),torch.randn(2,4,3,23,device='cuda')*.1,
      torch.randn(2,4,3,23,device='cuda')*.1,torch.randn(8,device='cuda')*.1,
      torch.full((8,),-3.,device='cuda')]
values=[[t.clone().requires_grad_() for t in base] for _ in range(2)]
u,dt,A,B,C,D,bias=values[0]
y=SelectiveScanCore.apply(u,dt,A,B,C,D,bias,True,1,1,True)
u,dt,A,B,C,D,bias=values[1];dt=torch.nn.functional.softplus(dt+bias[None,:,None])
B=B.repeat_interleave(2,dim=1);C=C.repeat_interleave(2,dim=1)
h=torch.zeros(2,8,3,device='cuda');ys=[]
for t in range(u.shape[-1]):
    h=(dt[:,:,t,None]*A[None]).exp()*h+dt[:,:,t,None]*B[:,:,:,t]*u[:,:,t,None]
    ys.append((h*C[:,:,:,t]).sum(-1)+D[None]*u[:,:,t])
ref=torch.stack(ys,-1)
torch.testing.assert_close(y,ref,atol=2e-5,rtol=2e-4)
weight=torch.randn_like(y);(y*weight).sum().backward();(ref*weight).sum().backward()
for actual,expected in zip(values[0],values[1]):
    torch.testing.assert_close(actual.grad,expected.grad,atol=2e-4,rtol=2e-3)
record={'scan_forward_max_error':float((y-ref).detach().abs().max()),'scan_gradients_passed':7,
        'kernel':'official selective_scan_cuda_core','gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__)}
model=YOLO(str(ROOT/'configs/Mamba-YOLO-B-visdrone.yaml')).model.cuda()
params=sum(v.numel() for v in model.parameters());record['params']=params
assert params==21800910,params
record['cgm_params']=18987187;record['relative_param_difference']=params/18987187-1
opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.937)
model.train();model.args=__import__('ultralytics.utils',fromlist=['get_cfg']).DEFAULT_CFG
batch={'img':torch.rand(2,3,960,960,device='cuda'),
       'batch_idx':torch.tensor([0,1],device='cuda'), 'cls':torch.tensor([[0.],[2.]],device='cuda'),
       'bboxes':torch.tensor([[.5,.5,.03,.05],[.3,.3,.05,.03]],device='cuda')}
torch.cuda.reset_peak_memory_stats();scaler=torch.amp.GradScaler('cuda')
for amp in [False,True]:
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',enabled=amp):loss,items=model(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    scaler.scale(loss.sum()).backward();scaler.unscale_(opt)
    grad=next(p.grad for n,p in model.named_parameters() if 'dt_projs_weight' in n)
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    scaler.step(opt);scaler.update()
    record['amp_loss' if amp else 'fp32_loss']=float(loss.detach().sum())
record['peak_gpu_gib_batch2']=torch.cuda.max_memory_allocated()/2**30
record['passed']=True
Path(a.out).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
