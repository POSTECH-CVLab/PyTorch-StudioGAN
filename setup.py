# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='StudioGAN',
    version='0.1.0',
    description='A Library for Experiment and Evaluation of GANs',
    long_description=readme,
    author='Minguk Kang and Jaesik Park',
    author_email='mgkang@postech.ac.kr',
    url='https://github.com/POSTECH-CVLab/PyTorch-StudioGAN',
    license=license,
    packages=find_packages(exclude=('configs', 'docs', 'figures'))
)
