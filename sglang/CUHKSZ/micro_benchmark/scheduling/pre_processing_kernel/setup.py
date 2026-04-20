from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="called_experts",
    ext_modules=[
        CUDAExtension(
            name="called_experts",
            sources=["binding.cpp", "called_experts_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
