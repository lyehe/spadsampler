from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="binomial_sampler",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for binomial sampling of image and video data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/binomial_sampler",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "opencv-python",
        "tifffile",
        "h5py",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)