from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

depends = [
    'include/mirage/persistent_kernel/tasks/argmax.cuh',
    'include/mirage/persistent_kernel/tasks/embedding.cuh',
    'include/mirage/persistent_kernel/tasks/linear.cuh',
    'include/mirage/persistent_kernel/tasks/norm_linear.cuh',
    'include/mirage/persistent_kernel/tasks/paged_attention.cuh',
    'include/mirage/persistent_kernel/tasks/reduction.cuh',
    'include/mirage/persistent_kernel/tasks/silu_mul_linear.cuh',
    'include/mirage/persistent_kernel/tasks/single_batch_decoding.cuh',
]

setup(
    name='runtime_kernel',
    ext_modules=[
        CUDAExtension(
            name='runtime_kernel',
            depends=depends,
            sources=[
                os.path.join(this_dir, 'runtime_kernel_wrapper.cu'),
            ],
            include_dirs=[
                os.path.join(this_dir, '../../include/mirage/persistent_kernel/tasks'),
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_80,code=sm_80',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)