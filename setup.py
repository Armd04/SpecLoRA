"""
Setup script for Speculative Decoding with Adaptive LoRA
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="speculative-lora",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Speculative Decoding with Adaptive LoRA Training for MLX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "mlx>=0.21.0",
        "mlx-lm>=0.20.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.1",
        "rich>=13.7.0",
        "numpy>=1.24.0",
        "click>=8.1.7",
        "transformers>=4.36.0",
        "huggingface-hub>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spec-lora=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
