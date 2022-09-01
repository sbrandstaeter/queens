"""QUEENS setup file for remote runs."""
from setuptools import setup

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
        "pymongo",
        "matplotlib",
        "plotly",
        "pandas",
        "vtk",
        "xarray",
        "scipy",
        "pyyaml",
        "pyfiglet",
    ],
)
