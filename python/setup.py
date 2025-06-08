
# setup.py
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "embeddedtensor",  # Module name in Python
        ["./bindings.cpp"],
        include_dirs=["src"],
        cxx_std=17,  # C++ standard, use 17 or higher if required
    ),
]

setup(
    name="embeddedtensor",
    version="0.1.0",
    author="Georgii Zagoruiko",
    description="A Python wrapper of NN layers for embedded systems support using pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    # packages=["your_package"],  # If you have Python-side code
    python_requires=">=3.6",
)