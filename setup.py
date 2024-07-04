#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='torch-kmeans',
    version='0.1.0',
    description='A PyTorch implementation of KMeans clustering.',
    author='Adrien Bolling',
    author_email='bolling.adrien@gmail.com',
    packages=find_packages(),
    python_requires='>=3.8',
    license="MIT",
    long_description=open('README.md').read(),
    install_requires=[
        'torch',
        'tqdm',
    ],
)