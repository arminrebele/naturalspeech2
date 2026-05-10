"""Build the monotonic_align Cython extension in place.

Run from this directory:
    python setup.py build_ext --inplace

The resulting `core.cpython-<abi>-<platform>.so` lands next to core.pyx and
is importable as `naturalspeech2.ops.monotonic_align.core`. The .so is
gitignored — the entrypoint script rebuilds it on first container start.
"""
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="core",
        sources=["core.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-fopenmp", "-O3"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="monotonic_align",
    ext_modules=cythonize(extensions, language_level=3),
)
