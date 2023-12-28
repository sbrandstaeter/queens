"""Rsync utils."""
from pathlib import Path

from queens.utils.run_subprocess import run_subprocess


def assemble_rsync_command(
    source,
    destination,
    archive=False,
    exclude=None,
    filters=None,
    verbose=True,
    rsh=None,
    host=None,
    rsync_options=None,
):
    """Assemble rsync command.

    Args:
        source (str, Path, list): paths to copy
        destination (str, Path): destination relative to host
        archive (bool): use the archive option
        exclude (str, list): options to exclude
        filters (str): filters for rsync
        verbose (bool): true for verbose
        rsh (str): remote ssh command
        host (str): host where to copy the files to
            rsync_options (list): additional rsync options

    Returns:
        str command to run rsync
    """

    def listify(obj):
        if isinstance(obj, (str, Path)):
            return [obj]
        return obj

    options = []
    if archive:
        options.append("--archive")
    if verbose:
        options.append("--verbose")
    if filters:
        options.append(f"--filter='{filters}'")
    if exclude:
        for e in listify(exclude):
            options.append(f"--exclude='{e}'")
    if rsync_options:
        options.extend(listify(rsync_options))
    if rsh:
        options.append(f"--rsh='{rsh}'")
    source = listify(source)
    if host:
        destination = f"{host}:{destination}"

    options = " ".join([str(option) for option in options])
    source = " ".join([str(file) for file in source])
    command = f"rsync {options} {source} {destination}"
    return command


def rsync(
    source,
    destination,
    archive=True,
    exclude=None,
    filters=None,
    verbose=True,
    rsh=None,
    host=None,
    rsync_options=None,
):
    """Run rsync command.

    Args:
        source (str, Path, list): paths to copy
        destination (str, Path): destination relative to host
        archive (bool): use the archive option
        exclude (str, list): options to exclude
        filters (str): filters for rsync
        verbose (bool): true for verbose
        rsh (str): remote ssh command
        host (str): host where to copy the files to
        rsync_options (list): additional rsync options
    """
    command = assemble_rsync_command(
        source=source,
        destination=destination,
        archive=archive,
        exclude=exclude,
        filters=filters,
        verbose=verbose,
        rsh=rsh,
        host=host,
        rsync_options=rsync_options,
    )

    run_subprocess(command)
