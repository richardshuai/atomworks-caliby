from setuptools import setup, find_packages

setup(
        name='CIFUtils: Python packages for reading mmCIF files from the PDB',
    version='2.0',
    packages=find_packages(),
    # We install open babel separately via `conda` (see README.md)
    install_requires=[
        'biotite>=0.40.0',
        'pytest>=8.2.0',
        'pandas>=1.4.2',
        'torch>=2.2.0',
        'numpy>=1.21.0',
    ]
)