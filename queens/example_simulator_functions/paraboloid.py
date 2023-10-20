"""2D paraboloid."""
# pylint: disable=invalid-name


def paraboloid(x1, x2, **_kwargs):
    """A paraboloid.

    See  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Args:
        x1 (float):  Input parameter 1
        x2 (float):  Input parameter 2

    Returns:
        float: Value of the function
    """
    a = 1.0
    b = 2.5
    return (x1 - a) ** 2 + (x2 - b) ** 2
