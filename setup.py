"""QUEENS setup file."""
import sys
from pathlib import Path

from setuptools import find_packages, setup


def read(fname):
    """Read text file.

    used e.g. to read the README.md and the requirements.txt files during setup.

    Args:
        fname (str | Path): File name to be read

    Returns:
        The content of the file fname
    """
    text_file = Path(__file__).parent / fname
    return text_file.read_text(encoding="utf-8")


def read_requirements(fname):
    """Load the requirement file `fname` and remove comments denoted by '#'.

    Args:
        fname (str): File name

    Returns:
        packages (list): List of the required packages
    """
    packages = []
    requirements_file = read(fname)
    for line in iter(requirements_file.splitlines()):
        requirement = line.partition('#')[0].rstrip()
        if requirement:
            packages.append(requirement)
    return packages


# QUEENS description
QUEENS_DESCRIPTION = (
    "A general purpose framework for Uncertainty Quantification, Physics-Informed Machine Learning"
    ", Bayesian Optimization, Inverse Problems and Simulation Analytics"
)

# Packages useful for developing and documentation not needed to run QUEENS
DEVELOPER_EXTRAS = [
    'pylint>=2.16',
    'pylint-exit',
    'isort>=5.0',
    'black==22.3.0',
    'pre-commit',
    'liccheck',
    'sphinx',
    'nbsphinx',
    'pandoc',
    'pip-tools',
    'vulture>=2.3',
]

# Keywords
KEYWORDS = (
    "Gaussian Processes, Uncertainty Quantification, Inverse analysis, Optimization, Sensitivity"
    "analysis, Multi-fidelity, Bayesian inference"
)

# Exit the installation process in case of incompatibility of the python version
REQUIRED_PYTHON_VERSION = '3.8'
SYSTEM_PYTHON_VERSION = f"{sys.version_info[0]}.{sys.version_info[1]}"
if SYSTEM_PYTHON_VERSION != REQUIRED_PYTHON_VERSION:
    MESSAGE = (
        f"\n\nYour python version is {SYSTEM_PYTHON_VERSION}, however QUEENS requires "
        f"{REQUIRED_PYTHON_VERSION}\n"
    )
    raise ImportError(MESSAGE)

# Actual setup process
setup(
    name="queens",
    version="1.1",
    author="QUEENS developers",
    description=(QUEENS_DESCRIPTION),
    keywords=KEYWORDS,
    packages=find_packages(exclude=["pqueens/tests"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"develop": DEVELOPER_EXTRAS},
    long_description=read('README.md'),
    setup_requires='pytest-runner',
    tests_require='pytest',
    entry_points={
        'console_scripts': [
            'queens = pqueens.main:main',
            'queens-build-singularity = pqueens.utils.cli_utils:build_singularity_cli',
            'queens-print-pickle = pqueens.utils.cli_utils:print_pickle_data_cli',
            'queens-build-html-coverage-report = '
            'pqueens.utils.cli_utils:build_html_coverage_report',
            'queens-remove-html-coverage-report = '
            'pqueens.utils.cli_utils:remove_html_coverage_report',
        ],
    },
)
