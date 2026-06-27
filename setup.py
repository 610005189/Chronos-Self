"""
Chronos-Self: A Self-Referential Continuous Dynamics System
Setup configuration for package installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from _version.py to keep in sync
import re
with open("chronos_core/_version.py", "r") as f:
    content = f.read()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    VERSION = match.group(1) if match else "0.1.1"

setup(
    name="chronos-self",
    version=VERSION,
    author="Chronos-Self Team",
    author_email="team@chronos-self.ai",
    description="A continuous time dynamics system with stable state representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/610005189/Chronos-Self",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchdiffeq>=0.2.3",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "transformers>=4.30.0",
        "chromadb>=0.4.0",
        "loguru>=0.7.0",
        "pydantic>=2.0.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chronos-train=chronos_core.training.cli:main",
            "chronos-validate=chronos_core.validation.cli:main",
        ],
    },
)