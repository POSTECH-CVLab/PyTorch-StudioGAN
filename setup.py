# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# setup.py

from setuptools import setup, find_packages



__version__ = '0.1'
url = 'https://github.com/POSTECH-CVLab/PyTorch-StudioGAN'

with open('README.md') as f:
    readme = f.read()


with open('LICENSE') as f:
    license = f.read()

install_requires = [
    'torch==1.6.0',
    'torchvision==0.7.0',
    'numpy',
    'matplotlib',
    'scikit-learn',
    'pandas',
    'scipy==1.1.0',
    'Pillow==6.2.2',
    'h5py',
    'urllib3',
    'tqdm',
]

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'mock']


setup(
    name='StudioGAN',
    version=__version__,
    description='A Library for Experiment and Evaluation of GANs',
    long_description=readme,
    author='Minguk Kang and Jaesik Park',
    author_email='mgkang@postech.ac.kr',
    url=url,
    license=license,
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
    include_package_data=True,
)
