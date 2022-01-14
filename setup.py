"""QUEENS setup file."""
import os
import sys
from shutil import copyfile

from setuptools import find_packages, setup


def read(fname):
    """Function to read the README file.

    Args:
        fname (str): File name to be read

    Returns:
        The content of the file fname
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_requirements(fname):
    """Load the requirement file `fname` and remove comments denoted by '#'.

    Args:
        fname (str): File name

    Returns:
        packages (list): List of the required packages
    """
    packages = []
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        for line in f:
            line = line.partition('#')[0].rstrip()
            if line:
                packages.append(line)
    return packages


# QUEENS description
queens_description = (
    "A general purpose framework for Uncertainty Quantification, Physics-Informed Machine Learning"
    ", Bayesian Optimization, Inverse Problems and Simulation Analytics"
)

# Packages useful for developing and documentation not needed to run QUEENS
developer_extras = [
    'pylint>=2.12',
    'pylint-exit',
    'isort>=5.0',
    'black==21.12b0',
    'pre-commit',
    'liccheck',
    'sphinx',
    'nbsphinx',
    'pandoc',
    'pip-tools',
    'vulture>=2.3',
]

# Keywords
keywords = (
    "Gaussian Processes, Uncertainty Quantification, Inverse analysis, Optimization, Sensitivity"
    "analysis, Multi-fidelity, Bayesian inference"
)

# Exit the installation process in case of incompatibility of the python version
required_python_version = '3.8'
system_python_version = "{}.{}".format(sys.version_info[0], sys.version_info[1])
if system_python_version != required_python_version:
    message = '\n\nYour python version is {}, however QUEENS requires {}\n'.format(
        system_python_version, required_python_version
    )
    raise ImportError(message)

# create a `main.py`` copy. This way QUEENS can be called directly through `queens` command
copyfile('pqueens/main.py', os.path.join(os.path.dirname(__file__), 'queens'))

# Actual setup process
setup(
    name="queens",
    version="1.1",
    author="QUEENS developers",
    description=(queens_description),
    keywords=keywords,
    scripts=["queens"],
    packages=find_packages(exclude=["pqueens/tests"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"develop": developer_extras},
    long_description=read('README.md'),
    setup_requires='pytest-runner',
    tests_require='pytest',
)
