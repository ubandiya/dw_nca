from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dw_nca",
    version="0.1.0",
    author="UBANDIYA NAJIB YUSUF",
    author_email="najibubandia@gmail.com",
    description="Distance-Weighted Neighbourhood Component Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ubandiya/dw_nca",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "scikit-learn>=0.22.0",
    ],
)
