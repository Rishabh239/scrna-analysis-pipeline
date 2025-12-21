"""
Setup script for scRNA-seq Analysis Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="scrna-analysis-pipeline",
    version="1.0.0",
    author="scRNA Analysis Pipeline",
    description="A comprehensive pipeline for single-cell RNA-seq analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scrna-analysis-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "joblib>=1.2.0",
        "leidenalg>=0.9.0",
    ],
    extras_require={
        "full": [
            "tensorflow>=2.10.0",
            "imbalanced-learn>=0.10.0",
            "gseapy>=1.0.0",
            "shap>=0.41.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "scrna-pipeline=src.cli:main",
        ],
    },
    include_package_data=True,
    keywords=["bioinformatics", "single-cell", "RNA-seq", "scanpy", "machine-learning"],
)
