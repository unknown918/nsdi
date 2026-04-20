from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="greedy_schedule",
    ext_modules=[
        CUDAExtension(
            name="greedy_schedule",
            sources=["greedy_schedule.cpp", "greedy_schedule_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
