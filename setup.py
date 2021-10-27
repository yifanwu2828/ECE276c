# setup.py
from setuptools import setup

setup(
    name='ail',
    version='0.1.0',
    packages=["common"],
    install_requires=['gym', 'torch', 'numpy'] 
)