from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "click>=8.1.3",
        "imageio>=2.33.0",
        "jax>=0.4.14",
        "joblib>=1.3.2",
        "matplotlib>=3.7.1",
        "mpi4py>=3.1.5",
        "numpyro>=0.12.0",
        "pandas>=2.0.3",
        "pyarrow>=14.0.1",
        "tqdm>=4.66.1",
    ],
)

__version__ = "0.1.6"

setup(
    name="flowfunnel",
    version=__version__,
    author="Yin Cheng",
    author_email="yin.sjtu@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yinsn/FlowFunnel",
    python_requires=">=3.8",
    description="Leveraging Bayesian hierarchical models to diagnose issues and proactively predict user flow stages in conversion funnels.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["flowfunnel=flowfunnel:run_flow_funnel"]},
    install_requires=install_requires,
    include_package_data=True,
)
