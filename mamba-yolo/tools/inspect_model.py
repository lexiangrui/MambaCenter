import json
from pathlib import Path
from common import ROOT
import torch,yaml
from copy import deepcopy
from ultralytics.nn.tasks import parse_model
torch.set_num_threads(2)
cfg=yaml.safe_load((ROOT/'configs/Mamba-YOLO-B-visdrone.yaml').read_text())
model,_=parse_model(deepcopy(cfg),ch=3,verbose=False)
params=sum(p.numel() for p in model.parameters())
record={'params':params,'nc':cfg['nc'],
        'scale':cfg['scales'],'source_commit':'51f7be9bd5ab907b6f7d2c91bb71ca5a4be063aa'}
(ROOT/'results/parameter_count.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
