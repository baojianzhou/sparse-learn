# -*- coding: utf-8 -*-

"""
This is a wrapper of head and tail projection. To generate sparse_module.so
file, please use the following command (suppose you have Linux/MacOS/MacBook):
    python setup.py build_ext --inplace
"""
import os
import numpy
from setuptools import setup
from setuptools import find_packages
from distutils.core import Extension

here = os.path.abspath(os.path.dirname(__file__))

src_files = ['sparse_learn/c/main_wrapper.c',
             'sparse_learn/c/head_tail_proj.c',
             'sparse_learn/c/fast_pcst.c',
             'sparse_learn/c/sort.c']
compile_args = ['-shared', '-Wall', '-g', '-O3', '-fPIC', '-std=c11', '-lm']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sparse_learn",
    version="0.1.1",
    author="Baojian Zhou",
    author_email="bzhou6@albany.edu",
    description="A package related with sparse learning methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    use_2to3=True,
    url="https://github.com/baojianzhou/sparse-learn",
    packages=find_packages(),
    install_requires=['numpy'],
    include_dirs=[numpy.get_include()],
    headers=['sparse_learn/c/head_tail_proj.h',
             'sparse_learn/c/fast_pcst.h',
             'sparse_learn/c/sort.h'],
    ext_modules=[
        Extension('sparse_learn',
                  sources=src_files,
                  language="C",
                  extra_compile_args=compile_args,
                  include_dirs=[numpy.get_include()])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", ],
    keywords='sparse learning, structure sparsity, head/tail projection')
