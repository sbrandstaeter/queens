"""QUEENS setup file for remote runs."""
import os

from setuptools import setup


def read(fname):
    """Function to read the README file.

    Args:
        fname (str): File name to be read

    Returns:
        The content of the file fname
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


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
    install_requires=[
        "docker",
        "netcdf4",
        "cython",
        "numpy",
        "pymongo==3.12.0",
        "matplotlib",
        "plotly",
        "pandas",
        "vtk",
        "xarray",
        "scipy",
    ],
)
