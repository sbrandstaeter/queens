"""CLI utils collection."""
import argparse

import pyfiglet

from pqueens.utils import ascii_art
from pqueens.utils.manage_singularity import create_singularity_image


def build_singularity_cli():
    """Build singularity image CLI wrapper."""
    ascii_art.print_crown()
    print(pyfiglet.figlet_format("SINBUILD", font="banner3-D"))
    parser = argparse.ArgumentParser(description="QUEENS singularity image builder utility")

    print("Building a singularity image! This might take some time")
    try:
        create_singularity_image()
        print("Done!")
    except Exception as cli_singularity_error:
        print("Building singularity failed!")
        raise cli_singularity_error
