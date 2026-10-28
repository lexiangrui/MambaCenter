# 上游源码与许可

本仓库保留第三方源码及其原始许可证。使用、修改和分发时遵循对应目录中的许可；本文件不重新授权第三方代码。

| 组件 | 目录 | 来源 / 固定版本 | 许可文件 |
|---|---|---|---|
| CenterNet（含本项目兼容补丁） | baseline/CenterNet | https://github.com/xingyizhou/CenterNet | baseline/CenterNet/LICENSE |
| DCNv2 | baseline/CenterNet/src/lib/models/networks/DCNv2 | CenterNet 原附带实现；运行时由 torchvision.ops.deform_conv2d 等价实现（tests/test_dcn.py） | 对应目录 LICENSE |
| Mamba selective scan | runtime/selective-scan/mamba | https://github.com/state-spaces/mamba ，10b5d6358f27966f6a40e4bf0baa17a460688128 | runtime/selective-scan/mamba/LICENSE |
| Mamba-YOLO / Ultralytics | mamba-yolo/vendor/Mamba-YOLO | https://github.com/HZAI-ZJNU/Mamba-YOLO ，51f7be9bd5ab907b6f7d2c91bb71ca5a4be063aa | mamba-yolo/vendor/Mamba-YOLO/LICENSE |

CUDA core 兼容和 FP32 评测补丁记录在 mamba-yolo/results/compatibility_patches.json；运行时来源记录在 runtime/selective-scan/provenance.json。
