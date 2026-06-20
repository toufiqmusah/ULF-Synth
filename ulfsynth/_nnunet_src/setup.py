from setuptools import setup, find_packages

setup(
    name='nnUNet_translation',
    version='0.1',
    packages=find_packages(include=['nnunetv2', 'nnunetv2.*']),
    include_package_data=True,
    install_requires=[],
)
