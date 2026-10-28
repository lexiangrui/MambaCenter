"""Build official Mamba v2.2.6.post3 CUDA scan for the target GPU.
Only architecture selection/build plumbing differs from upstream setup.py.
"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
root=Path(__file__).resolve().parent/'mamba'
files=['selective_scan.cpp','selective_scan_fwd_fp32.cu','selective_scan_fwd_fp16.cu',
       'selective_scan_fwd_bf16.cu','selective_scan_bwd_fp32_real.cu',
       'selective_scan_bwd_fp32_complex.cu','selective_scan_bwd_fp16_real.cu',
       'selective_scan_bwd_fp16_complex.cu','selective_scan_bwd_bf16_real.cu',
       'selective_scan_bwd_bf16_complex.cu']
setup(name='centermamba-selective-scan',version='0.1',ext_modules=[CUDAExtension(
    name='selective_scan_cuda',sources=[str(root/'csrc'/'selective_scan'/f) for f in files],
    include_dirs=[str(root/'csrc'/'selective_scan')],
    extra_compile_args={'cxx':['-O3','-std=c++17'],
        'nvcc':['-O3','-std=c++17','--use_fast_math','--expt-relaxed-constexpr',
                '--expt-extended-lambda','-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_OPERATORS__',
                '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_BFLOAT162_OPERATORS__',
                '--threads','2']})],cmdclass={'build_ext':BuildExtension})
