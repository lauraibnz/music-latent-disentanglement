from setuptools import setup, find_packages

setup(
    name="mld",
    version="0.1.0",
    description="Music latent disentanglement for controllable generation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
