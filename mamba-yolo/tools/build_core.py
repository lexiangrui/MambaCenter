"""Build the official v2 core extension; PyTorch selects the target CUDA architecture."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

root = Path(__file__).resolve().parents[1]
src = root / 'vendor/Mamba-YOLO/selective_scan/csrc/selective_scan'
setup(name='mbyolo-scan-core', ext_modules=[CUDAExtension(
    name='selective_scan_cuda_core',
    sources=[str(src/'cus'/f) for f in ('selective_scan.cpp', 'selective_scan_core_fwd.cu', 'selective_scan_core_bwd.cu')],
    include_dirs=[str(src)],
    extra_compile_args={'cxx':['-O3','-std=c++17'], 'nvcc':[
        '-O3','-std=c++17','--use_fast_math','--expt-relaxed-constexpr','--expt-extended-lambda',
        '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT162_OPERATORS__','-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
        '--threads','2']})],
    cmdclass={'build_ext':BuildExtension})
