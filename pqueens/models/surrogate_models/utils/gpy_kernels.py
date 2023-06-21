"""Kernels for GP regression with GPy."""

import GPy
import numpy as np


def get_gpy_kernel_type(kernel_type):
    """Choose kernel setup method by kernel type.

    Args:
        kernel_type (str): Kernel type to setup

    Returns:
        function object (obj): Function object for implementation type of setup of kernel
    """
    return {
        "prod_rbf": lambda: setup_prod_rbf,
        "sum_rbf": lambda: setup_sum_rbf,
        "prod_matern": lambda: setup_prod_matern,
        "sum_matern": lambda: setup_sum_matern,
        "rbf": lambda: setup_rbf,
        "matern": lambda: setup_matern,
    }.get(kernel_type, lambda: ValueError("Unknown kernel."))()


def setup_rbf(input_dim, variance_0, lengthscale_0, ard):
    r"""Radial Basis Function (RBF) kernel.

     Aka squared-exponential or Gaussian kernel.

    :math:`k(r) = \sigma^2 \exp \bigg(- \frac{1}{2} r^2 \bigg)`

    Args:
        input_dim (int): Number of input dimensions
        variance_0 (float): Initial variance
        lengthscale_0 (float): Initial lengthscale
        ard (bool): Automatic relevant determination turned on if *True*

    Returns:
        kernel (GPy.kern.scr.rbf.RBF object): GP kernel
    """
    active_dimensions = np.array(range(input_dim))
    kernel = GPy.kern.RBF(
        input_dim=input_dim,
        variance=variance_0,
        lengthscale=lengthscale_0,
        ARD=ard,
        active_dims=active_dimensions,
    )
    return kernel


def setup_matern(input_dim, variance_0, lengthscale_0, ard):
    """Matern 5/2 kernel.

    Args:
        input_dim (int): Number of input dimensions
        variance_0 (float): Initial variance
        lengthscale_0 (float): Initial lengthscale
        ard (bool): Automatic relevant determination turned on if *True*

    Returns:
        kernel (GPy.kern.scr.stationary.Matern52 object): GP kernel
    """
    active_dimensions = np.array(range(input_dim))
    kernel = GPy.kern.Matern52(
        input_dim=input_dim,
        variance=variance_0,
        lengthscale=lengthscale_0,
        ARD=ard,
        active_dims=active_dimensions,
    )
    return kernel


def setup_sum_matern(input_dim, variance_0, lengthscale_0, ard):
    """Sum of Matern 5/2 kernels.

    Args:
        input_dim (int): Number of input dimensions
        variance_0 (float): Initial variance
        lengthscale_0 (float): Initial lengthscale
        ard (bool): automatic Relevant determination turned on if *True*

    Returns:
        kernel (GPy.kern.scr.add.Add object): GP kernel
    """
    k_list = [
        GPy.kern.Matern52(
            input_dim=1,
            variance=variance_0,
            lengthscale=lengthscale_0,
            ARD=ard,
            active_dims=[dim],
        )
        for dim in range(input_dim)
    ]
    kernel = k_list[0]
    if len(k_list) > 1:
        for k_ele in k_list[1:]:
            kernel += k_ele
    return kernel


def setup_prod_matern(input_dim, variance_0, lengthscale_0, ard):
    """Product of Matern 5/2 kernels.

    Args:
        input_dim (int): Number of input dimensions
        variance_0 (float): Initial variance
        lengthscale_0 (float): Initial lengthscale
        ard (bool): Automatic relevant determination turned on if *True*

    Returns:
        kernel (GPy.kern.scr.prod.Prod object): GP kernel
    """
    k_list = [
        GPy.kern.Matern52(
            input_dim=1,
            variance=variance_0,
            lengthscale=lengthscale_0,
            ARD=ard,
            active_dims=[dim],
        )
        for dim in range(input_dim)
    ]
    kernel = k_list[0]
    if len(k_list) > 1:
        for k_ele in k_list[1:]:
            kernel *= k_ele
    return kernel


def setup_sum_rbf(input_dim, variance_0, lengthscale_0, ard):
    """Sum of RBF kernels.

    Args:
        input_dim (int): Number of input dimensions
        variance_0 (float): Initial variance
        lengthscale_0 (float): Initial lengthscale
        ard (bool): Automatic relevant determination turned on if *True*

    Returns:
        kernel (GPy.kern.scr.add.Add object): GP kernel
    """
    k_list = [
        GPy.kern.RBF(
            input_dim=1,
            variance=variance_0,
            lengthscale=lengthscale_0,
            ARD=ard,
            active_dims=[dim],
        )
        for dim in range(input_dim)
    ]
    kernel = k_list[0]
    if len(k_list) > 1:
        for k_ele in k_list[1:]:
            kernel += k_ele
    return kernel


def setup_prod_rbf(input_dim, variance_0, lengthscale_0, ard):
    """Product of RBF kernels.

    Args:
        input_dim (int): Number of input dimensions
        variance_0 (float): Initial variance
        lengthscale_0 (float): Initial lengthscale
        ard (bool): Automatic relevant determination turned on if True

    Returns:
        kernel (GPy.kern.scr.prod.Prod object): GP kernel
    """
    k_list = [
        GPy.kern.RBF(
            input_dim=1,
            variance=variance_0,
            lengthscale=lengthscale_0,
            ARD=ard,
            active_dims=[dim],
        )
        for dim in range(input_dim)
    ]
    kernel = k_list[0]
    if len(k_list) > 1:
        for k_ele in k_list[1:]:
            kernel *= k_ele
    return kernel
