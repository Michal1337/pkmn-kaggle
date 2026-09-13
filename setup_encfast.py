"""Build the Cython encode fast path (rl/_encfast.pyx -> rl/_encfast*.so).

    PYTHONPATH=<dir-with-Cython> python setup_encfast.py build_ext --inplace

Needs Cython + numpy at BUILD time; the resulting .so needs NEITHER at runtime (rl/encoding.py
imports it opportunistically and falls back to pure Python when it's absent). The .so is
platform/ABI-specific -- build it on the training node, do not commit it.
"""
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    name="encfast",
    ext_modules=cythonize(
        [Extension("rl._encfast", ["rl/_encfast.pyx"], include_dirs=[np.get_include()])],
        compiler_directives={"language_level": "3"},
        quiet=True,
    ),
)
