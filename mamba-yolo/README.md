# Mamba-YOLO-B

保留[官方Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)的B结构，commit `51f7be9bd5ab907b6f7d2c91bb71ca5a4be063aa`：width0.50、depth0.33、10个VisDrone类别，总参数21,800,910。
不使用之前已取消的宽度0.46变体。

## 当前状态与结果

已完成75/300轮，用户于2026-10-08取消训练。第70轮FP32 EMA权重已保存；没有完整300轮结果。

| 验证轮数 | AP (%) | AP50 (%) |
|---|---:|---:|
| 70 | 30.17 | 49.27 |
| 75 | 30.51 | 49.65 |

结果索引和比较限制见 [项目实验说明](../docs/experiments.md)。
第70轮使用300轮学习率计划；它与CenterNet系模型在增强、优化器、预训练、损失和EMA上不同，属于整体检测器比较。

## 数据、训练与评测

VisDrone train6471 / val548 / 10类。YOLO标签由共用COCO标注转换，图片通过软链复用原始数据。数据位置由 --source 和 --out 参数指定。

输入960，scratch初始化，SGD lr0=.01、momentum=.937、weight_decay=.0005，3轮warmup、AMP、seed317。
Microbatch2、nominal batch16，梯度累积沿用官方warmup行为；BatchNorm看到microbatch。
每5轮完整验证；max500/image、conf=.01、NMS IoU=.7、FP32原图坐标COCO maxDets=[1,10,100]。主要报告整体AP/AP50。

## 运行入口

先安装 Python 依赖，并在目标 CUDA 机器构建独立的 core 扩展：

```bash
# 仓库根目录
source .venv/bin/activate
export MAX_JOBS=2
python mamba-yolo/tools/build_core.py build_ext \
  --build-lib mamba-yolo/.deps --build-temp mamba-yolo/build/core
export PYTHONPATH="$PWD/mamba-yolo/.deps${PYTHONPATH:+:$PYTHONPATH}"

python mamba-yolo/tools/prepare_data.py --source data/VisDrone2019-DET --out data/visdrone-yolo
mkdir -p runs
python mamba-yolo/tools/validate.py --out runs/mamba-yolo-check.json
python mamba-yolo/tools/train.py --data data/visdrone-yolo/dataset.yaml \
  --out runs/mamba-yolo-new --epochs 300 --batch 2
```

新训练应在原生扫描和梯度检查通过后启动。
GPU 架构由 PyTorch 自动选择，也可自行设置 TORCH_CUDA_ARCH_LIST；core 扩展不使用 runtime/selective-scan 的内核替代。

评测、汇总与显式续训示例：

```bash
python mamba-yolo/tools/evaluate.py --ckpt runs/mamba-yolo-new/train/weights/epoch70_fp32.pt \
  --data data/visdrone-yolo/dataset.yaml --out runs/mamba-yolo-eval \
  --reference-summary results/comparison.json
python mamba-yolo/tools/summarize.py --out runs/mamba-yolo-new --reference-summary results/comparison.json
python mamba-yolo/tools/train.py --data data/visdrone-yolo/dataset.yaml \
  --out runs/mamba-yolo-new --resume runs/mamba-yolo-new/train/weights/last.pt --batch 2
```

每个新训练选择新的 --out；续训必须显式指定 --resume，不自动恢复已停止实验。
源码兼容修改记录见 [results/compatibility_patches.json](results/compatibility_patches.json)，vendor 保留上游许可。GPU 检查报告、机器配置、权重和冻结快照不随仓库发布。
