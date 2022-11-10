from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="ML_project for homework1 of MADE_MLOps_2022",
    author="Roman Zilotov",
    entry_points={
        "console_scripts": [
            "launch_data_loader = src.data.data_loader:main",
            "make_synthetic_data = tests.make_synthetic_data:main"
        ]
    },
    install_requires=required,
    license="MIT",
)