from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import platform

if __name__ == "__main__":
    ext_modules = []
    if torch.cuda.is_available():
        cuda_extension = CUDAExtension(
            name="sp_avg.cuda",
            sources=[
                "sp_avg/cuda/gather.cpp",
                "sp_avg/cuda/gather_kernel.cu",
                "sp_avg/cuda/scatter.cpp",
                "sp_avg/cuda/scatter_kernel.cu",
                "sp_avg/cuda/common_cuda.cu",
                "sp_avg/cuda/pybind_cuda.cpp",
                "sp_avg/common.cpp",
            ],
        )
        ext_modules.append(cuda_extension)

    setup(
        name="sp_avg",
        author="Yang",
        ext_modules=ext_modules,
        packages=find_packages(),
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        install_requires=["torch>=1.7"],
        url="",
        description="Spatially Incremental Generative Engine (SIGE) with scatter average",
        long_description="",  # 可以添加项目的详细描述
        long_description_content_type="text/markdown",
        version='0.1',
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
    )
