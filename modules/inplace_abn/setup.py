import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])

if TORCH_MAJOR < 1:
    raise RuntimeError("Inplace ABN requires Pytorch 1.0 or newer.\n" +
                       "The latest stable release can be obtained from https://pytorch.org/")
version = '0.1.2'
setup(
    name='inplace_abn',
    version=version,
    ext_modules=[
        CUDAExtension('inplace_abn', [
            'src/inplace_abn.cpp',
            'src/inplace_abn_cpu.cpp',
            'src/inplace_abn_cuda.cu',
            'src/inplace_abn_cuda_half.cu',
        ], extra_compile_args={'cxx': ['-O3', ],
                               'nvcc': ['-O3',
                                        "-DCUDA_HAS_FP16=1",
                                        '--expt-extended-lambda',
                                        '--use_fast_math']})
    ],
    description='PyTorch Extensions: Inplace ABN',
    cmdclass={
        'build_ext': BuildExtension
    })
