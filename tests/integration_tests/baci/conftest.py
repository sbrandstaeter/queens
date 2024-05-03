"""Pytest configuration for BACI integration tests."""

import pytest


@pytest.fixture(name="setup_symbolic_links_baci", autouse=True)
def fixture_setup_symbolic_links_baci(baci_link_paths, baci_source_paths_for_gitlab_runner):
    """Set-up of BACI symbolic links.

    Args:
        baci_link_paths (Path): destination for symbolic links to executables
        baci_source_paths_for_gitlab_runner (Path): Several paths that are needed to build symbolic
                                                links to executables
    """
    (
        dst_baci_release,
        dst_post_ensight,
        dst_post_processor,
    ) = baci_link_paths

    (
        src_baci_release,
        src_post_ensight,
        src_post_processor,
    ) = baci_source_paths_for_gitlab_runner
    # check if symbolic links are existent
    try:
        # create link to default baci-release location if no link is available
        if not dst_baci_release.is_symlink():
            if not src_baci_release.is_file():
                raise FileNotFoundError(
                    f'Failed to create link to default baci-release location.\n'
                    f'No baci-release found under default location:\n'
                    f'\t{src_baci_release}\n'
                )
            dst_baci_release.symlink_to(src_baci_release)
        # create link to default post_ensight location if no link is available
        if not dst_post_ensight.is_symlink():
            if not src_post_ensight.is_file():
                raise FileNotFoundError(
                    f'Failed to create link to default post_ensight location.\n'
                    f'No post_ensight found under default location:\n'
                    f'\t{src_post_ensight}\n'
                )
            dst_post_ensight.symlink_to(src_post_ensight)
        # create link to default post_processor location if no link is available
        if not dst_post_processor.is_symlink():
            if not src_post_processor.is_file():
                raise FileNotFoundError(
                    f'Failed to create link to default post_processor location.\n'
                    f'No post_processor found under default location:\n'
                    f'\t{src_post_processor}\n'
                )
            dst_post_processor.symlink_to(src_post_processor)

        # check if existing link to baci-release works and points to a valid file
        if not dst_baci_release.resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_baci_release}\n'
                f'It points to (non-existing): {dst_baci_release.resolve()}\n'
            )
        # check if existing link to post_ensight works and points to a valid file
        if not dst_post_ensight.resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_post_ensight}\n'
                f'It points to: {dst_post_ensight.resolve()}\n'
            )
        # check if existing link to post_processor works and points to a valid file
        if not dst_post_processor.resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_post_processor}\n'
                f'It points to: {dst_post_processor.resolve()}\n'
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            'Please make sure to make the missing executable available under the given '
            'path OR\n'
            'make sure the symbolic link under the config directory points to an '
            'existing file! \n'
            'You can create the necessary symbolic links on Linux via:\n'
            '-------------------------------------------------------------------------\n'
            'ln -s <path/to/baci-release> <QUEENS_BaseDir>/config/baci-release\n'
            'ln -s <path/to/post_ensight> <QUEENS_BaseDir>/config/post_ensight\n'
            'ln -s <path/to/post_processor> <QUEENS_BaseDir>/config/post_processor\n'
            '-------------------------------------------------------------------------\n'
            '...and similar for the other links.'
        ) from error
