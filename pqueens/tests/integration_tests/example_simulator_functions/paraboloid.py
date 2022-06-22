"""2D paraboloid."""
# pylint: disable=invalid-name


def paraboloid(x1, x2):
    """A paraboloid.

    see  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Args:
        x1 (float):  Input one
        x2 (float):  Input two

    Returns:
        float: Value of function
    """
    a = 1.0
    b = 2.5
    return (x1 - a) ** 2 + (x2 - b) ** 2
