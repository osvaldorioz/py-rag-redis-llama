from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pybind11

#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#python3.12 setup.py build_ext --inplace

#find / -name "libc10.so" 2>/dev/null
#export LD_LIBRARY_PATH=/home/hadoop/Documentos/cpp_programs/pybind/py-rag-redis/myenv/lib/python3.12/site-packages/torch/lib
#echo $LD_LIBRARY_PATH

setup(
    name="rag_module",
    ext_modules=[
        CppExtension(
            "rag_module",
            ["bindings.cpp"],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=["-std=c++17", "-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)