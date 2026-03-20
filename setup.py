from setuptools import setup, find_packages

setup(
    name="osa-adapt",
    version="1.0.0",
    description="OSA-Adapt: Severity-Aware Domain Adaptation for Clinical Sleep Staging",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="PSG Research Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "physioex": ["physioex"],
        "dev": ["pytest>=7.0", "hypothesis>=6.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
