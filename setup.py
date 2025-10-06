from setuptools import setup, find_packages

setup(
    name="musdis",
    version="0.1.0",
    description="Music Disentanglement for Controllable Generation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)