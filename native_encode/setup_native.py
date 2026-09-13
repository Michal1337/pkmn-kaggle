"""Build the M3 native encode Cython module: rl/native_encode/encode_native.pyx -> encode_native*.so

    PYTHONPATH=<dir-with-Cython> python setup_native.py build_ext --inplace

Same build model as setup_encfast.py (Cython at build time only; the .so needs neither Cython nor
the engine at runtime). Byte-identical to rl/encoding.py::TokenEncoder.encode; see README.md.
"""
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    name="encode_native",
    ext_modules=cythonize(
        [Extension("encode_native", ["encode_native.pyx"], include_dirs=[np.get_include()])],
        compiler_directives={"language_level": "3"},
        quiet=True,
    ),
)
