from setuptools import setup, find_packages

setup(
name='benchmarking_framework',
version='0.1.0',
description='A framework for benchmarking optimization algorithms against multidimensional black-box functions.',
author='Ilya Komarov',
author_email='ilya.komarov@henkel.com',
packages=find_packages(),
install_requires=[
'numpy',
'pandas',
'pymc',
'aesara',
'scipy',
'botorch',
'torch',
'gpytorch',
'deepdiff',
'multiprocess',
'ipykernel',
'xgboostlss'
],
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
)