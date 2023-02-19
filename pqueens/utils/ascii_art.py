"""ASCII art module."""
import logging

import pyfiglet

_logger = logging.getLogger(__name__)


def print_bmfia_acceleration():
    """Print BMFIA rocket."""
    rocket = r"""
          !
          !
          ^
         / \
       / ___ \
      |=     =|
      |       |
      |-BMFIA-|
      |       |
      |       |
      |       |
      |       |
      |       |
      |       |
      |       |
     /| ##!## | \
   /  | ##!## |   \
 /    | ##!## |     \
|     / ^ | ^  \     |
|    /   (|)    \    |
|   /    (|)     \   |
|  /   ((   ))    \  |
| /     ((:))      \ |
|/      ((:))       \|
       ((   ))
        (( ))
         ( )
          .
          .
          .
    """
    print_centered_multiline_block(rocket)


def print_crown(output_width=63):
    """Print crown.

    Args:
        output_width (int): Terminal output width (default is 63)
    """
    crown = r"""
        *
      * | *
     * \|/ *
* * * \|O|/ * * *
 \o\o\o|O|o/o/o/
 (<><><>O<><><>)
  '==========='
    """
    print_centered_multiline_block(crown, output_width)


def print_banner(message="QUEENS", output_width=63):
    """Print banner.

    Args:
        message (str): Message in banner
        output_width (int): Terminal output width (default is 63)
    """
    print_centered_multiline_block(pyfiglet.figlet_format(message, font="banner3-D"), output_width)


def print_centered_multiline_block(string, output_width=63):
    """Print a multiline text in the center as a block.

    Args:
        string (str): String to be printed
        output_width (int): Terminal output width (default is 63)
    """
    lines = string.split("\n")
    max_line_width = max(len(line) for line in lines)
    if max_line_width % 2:
        output_width += 1
    for line in lines:
        _logger.info(line.ljust(max_line_width).center(output_width))


def print_centered_multiline(string, output_width=63):
    """Center every line of a multiline text.

    Args:
        string (str): String to be printed
        output_width (int): Terminal output width (default is 63)
    """
    lines = string.split("\n")
    for line in lines:
        _logger.info(line.strip().center(output_width))


def print_banner_and_description():
    """Print banner and the description."""
    print_crown()
    print_banner()
    description = """
    A general purpose framework for Uncertainty Quantification,
    Physics-Informed Machine Learning, Bayesian Optimization,
    Inverse Problems and Simulation Analytics
    """
    print_centered_multiline(description)
