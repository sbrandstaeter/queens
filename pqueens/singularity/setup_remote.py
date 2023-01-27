"""QUEENS setup file for remote runs."""
from pathlib import Path

from setuptools import setup


def read_requirements(fname):
    """Load the requirement file `fname` and remove comments denoted by '#'.

    Args:
        fname (str): File name

    Returns:
        packages (list): List of the required packages
    """
    packages = []
    with open(str(Path(Path(__file__).parent, fname)), encoding="utf-8") as f:
        for line in f:
            line = line.partition('#')[0].rstrip()
            if line:
                packages.append(line)
    return packages


setup(
    name="pqueens-remote",
    version="1.1",
    description=("A package for Uncertainty Quantification and Bayesian optimization"),
    keywords="Gaussian Processes, Uncertainty Quantification",
    packages=[
        'pqueens',
        'pqueens.database',
        'pqueens.drivers',
        'pqueens.utils',
        'pqueens.external_geometry',
        'pqueens.randomfields',
    ],
    install_requires=read_requirements("requirements.txt"),
)
