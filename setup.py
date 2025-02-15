from setuptools import setup, find_packages

setup(
    name="amino_acid_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
    ],
)
