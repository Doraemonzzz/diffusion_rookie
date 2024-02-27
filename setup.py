from setuptools import find_packages, setup

setup(
    name="diffusion_rookie",
    packages=find_packages(
        exclude=[
            "tests",
            "benchmarks",
        ]
    ),
    install_requires=["torch", "einops", "triton"],
    version="0.0.0",
    author="Doraemonzzz",
    include_package_data=True,
)
