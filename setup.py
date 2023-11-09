from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "daft>=0.1.2",
        "pymc>=5.6.1",
        "pytensor>=2.12.3",
        "tqdm>=4.66.1",
    ],
)

__version__ = "0.0.7"

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
    install_requires=install_requires,
    include_package_data=True,
)
