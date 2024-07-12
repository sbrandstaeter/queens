"""ASCII art module."""

import logging

import pyfiglet

from queens.utils.print_utils import DEFAULT_OUTPUT_WIDTH

_logger = logging.getLogger(__name__)


def print_bmfia_acceleration(output_width=DEFAULT_OUTPUT_WIDTH):
    """Print BMFIA rocket.

    Args:
        output_width (int): Terminal output width
    """
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
    print_centered_multiline_block(rocket, output_width)


def print_classification():
    """Print like a sir as the iterator is classyfication."""
    las = """                    ./@@@@@@@@
             *&@@@@@@@@@@@@@@@@(
        #@@@@@@@@@@@@@@@@@@@@@@@@
    /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   ,@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.
     #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#       ,#@@@@@@@.
       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&@@@@@@@@@@@#
        /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&.
          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
           ,@@@@@@@@@@@@@@@@@@.              *@
         ,@@@@@@@@@@@@@&        (@@@@@@@@@@@&  *%
  .#@@@@@@@@@@@@@@%    ,(###@      @*.%%/@@&&   /@
#@@@@@@@,   *@@&   *@ @*  ,@@@     /&      @     @     ,@&
 .          @@*  *@, @.  , ( ,(       %&@&,&@@@@@@@@@@@@@&
           #@& .@@@@@   #    ,(   /@@@@@@@@@@/    @
           @@,  #  /@  ,&  %,@  *@@@          @.  @
          ,@@*  &      *%%#    %@@   @& %,**   , %#
           @@@  *,          @@@@.     .         ,@
           ,@@@  #                             (@                ,*
            ,@@@,%                           ,@%              &@. &@.
              .@@@&                       .@@#              .@
                  &@@@@,             *@@@%% @@             %@
                      %@@,,#@    & /@@ /@%@ %@@@@@&/     /@/
                        .@   &%  .@@@@  @//, *@@.   %@@@@/
                         /@/   ., @  #@@@@  @/# @.
                        *@@@#   % %  #@(  @  &.  @/
                       %@  @@    .@% #@(   @ *.   @.
                      .@   &@       &,&@*  @#@,    @
                      @*  ,@@@@@@%##, #  ,(  @     #/

                            C L A S S ification
    """
    print_centered_multiline_block(las)


def print_crown(output_width=DEFAULT_OUTPUT_WIDTH):
    """Print crown.

    Args:
        output_width (int): Terminal output width
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


def print_points_iterator(output_width=DEFAULT_OUTPUT_WIDTH):
    """Print points iterator.

    Args:
        output_width (int): Terminal output width
    """
    points = r"""
    @@@@@@@@@@@
  @@@        @@@                                           @@@@@@@@@@@
  @@@         @@@                                        @@@@       @@@@
  @@@        @@@                                         @@@         @@@
   @@@@@@,@@@@@                #@@@@@@@@@                @@@         @@@
    @@@@@@@@@@               @@@@      @@@@               @@@@@@@@@@@@@
      @@@@@@@                @@@         @@@                @@@@@@@@@@
       @@@@                  @@@        @@@                  @@@@@@@
        @@                    @@@      @@@                     @@@
                               @@@@@@@@@@                       @
                                @@@@@@@@
                                 @@@@@@
                                   @@
    """
    print_centered_multiline_block(points, output_width)


def print_banner(message="QUEENS", output_width=DEFAULT_OUTPUT_WIDTH):
    """Print banner.

    Args:
        message (str): Message in banner
        output_width (int): Terminal output width
    """
    print_centered_multiline_block(pyfiglet.figlet_format(message, font="banner3-D"), output_width)


def print_centered_multiline_block(string, output_width=DEFAULT_OUTPUT_WIDTH):
    """Print a multiline text in the center as a block.

    Args:
        string (str): String to be printed
        output_width (int): Terminal output width
    """
    lines = string.split("\n")
    max_line_width = max(len(line) for line in lines)
    if max_line_width % 2:
        output_width += 1
    for line in lines:
        _logger.info(line.ljust(max_line_width).center(output_width))


def print_centered_multiline(string, output_width=DEFAULT_OUTPUT_WIDTH):
    """Center every line of a multiline text.

    Args:
        string (str): String to be printed
        output_width (int): Terminal output width
    """
    lines = string.split("\n")
    for line in lines:
        _logger.info(line.strip().center(output_width))


def print_banner_and_description(output_width=DEFAULT_OUTPUT_WIDTH):
    """Print banner and the description.

    Args:
        output_width (int): Terminal output width
    """
    print_crown()
    print_banner()
    description = """
    A general purpose framework for Uncertainty Quantification,
    Physics-Informed Machine Learning, Bayesian Optimization,
    Inverse Problems and Simulation Analytics
    """
    print_centered_multiline(description, output_width)
