# Selective-scan 运行时

共享底层运行时，与检测器源码分开存放。

- mamba/：Mamba v2.2.6.post3 上游源码，commit 10b5d6358f27966f6a40e4bf0baa17a460688128；保留已有兼容补丁和上游许可证。
- packages/：目标机器构建的 selective_scan_cuda；由 Git 忽略，不发布平台二进制。
- build.py：仅构建原生 selective_scan_cuda，不更改 SSM 数学。

在配置好的 CUDA 开发环境中，从仓库根目录执行：

```bash
source .venv/bin/activate
export MAX_JOBS=2
# PyTorch 根据可见 GPU 选择架构；无可见 GPU 时自行设置 TORCH_CUDA_ARCH_LIST
python runtime/selective-scan/build.py build_ext \
  --build-lib runtime/selective-scan/packages --build-temp runtime/selective-scan/build
export PYTHONPATH="$PWD/runtime/selective-scan/packages:$PWD/runtime/selective-scan/mamba${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; import selective_scan_cuda; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn'
```

Python 辅助依赖安装到虚拟环境。mambacenter 的 PYTHONPATH 同时包含 packages 和 mamba；Mamba-YOLO 独立构建 selective_scan_cuda_core，见其 README。
不同服务器需重新编译并验证，GPU 调度和 CUDA 环境由使用者配置。
