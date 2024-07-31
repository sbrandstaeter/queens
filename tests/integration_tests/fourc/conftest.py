"""Pytest configuration for fourc integration tests."""

import pytest


@pytest.fixture(name="setup_symbolic_links_fourc", autouse=True)
def fixture_setup_symbolic_links_fourc(fourc_link_paths, fourc_build_paths_for_gitlab_runner):
    """Set-up of fourc symbolic links.

    Args:
        fourc_link_paths (Path): destination for symbolic links to executables
        fourc_build_paths_for_gitlab_runner (Path): Several paths that are needed to build symbolic
                                                links to executables
    """
    (
        dst_fourc,
        dst_post_ensight,
        dst_post_processor,
    ) = fourc_link_paths

    (
        fourc,
        post_ensight,
        post_processor,
    ) = fourc_build_paths_for_gitlab_runner
    # check if symbolic links are existent
    try:
        # create link to default fourc executable location if no link is available
        if not dst_fourc.is_symlink():
            if not fourc.is_file():
                raise FileNotFoundError(
                    f"Failed to create link to default fourc location.\n"
                    f"No fourc found under default location:\n"
                    f"\t{fourc}\n"
                )
            dst_fourc.symlink_to(fourc)
        # create link to default post_ensight location if no link is available
        if not dst_post_ensight.is_symlink():
            if not post_ensight.is_file():
                raise FileNotFoundError(
                    f"Failed to create link to default post_ensight location.\n"
                    f"No post_ensight found under default location:\n"
                    f"\t{post_ensight}\n"
                )
            dst_post_ensight.symlink_to(post_ensight)
        # create link to default post_processor location if no link is available
        if not dst_post_processor.is_symlink():
            if not post_processor.is_file():
                raise FileNotFoundError(
                    f"Failed to create link to default post_processor location.\n"
                    f"No post_processor found under default location:\n"
                    f"\t{post_processor}\n"
                )
            dst_post_processor.symlink_to(post_processor)

        # check if existing link to fourc works and points to a valid file
        if not dst_fourc.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {dst_fourc}\n"
                f"It points to (non-existing): {dst_fourc.resolve()}\n"
            )
        # check if existing link to post_ensight works and points to a valid file
        if not dst_post_ensight.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {dst_post_ensight}\n"
                f"It points to: {dst_post_ensight.resolve()}\n"
            )
        # check if existing link to post_processor works and points to a valid file
        if not dst_post_processor.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {dst_post_processor}\n"
                f"It points to: {dst_post_processor.resolve()}\n"
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Please make sure to make the missing executable available under the given "
            "path OR\n"
            "make sure the symbolic link under the config directory points to an "
            "existing file! \n"
            "You can create the necessary symbolic links on Linux via:\n"
            "-------------------------------------------------------------------------\n"
            "ln -s <path/to/fourc> <QUEENS_BaseDir>/config/fourc\n"
            "ln -s <path/to/post_ensight> <QUEENS_BaseDir>/config/post_ensight\n"
            "ln -s <path/to/post_processor> <QUEENS_BaseDir>/config/post_processor\n"
            "-------------------------------------------------------------------------\n"
            "...and similar for the other links."
        ) from error
