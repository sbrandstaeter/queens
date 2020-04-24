import subprocess


def run_subprocess(command_string):
    """
    Run a system command outside of the Python script anc check for errors and
    stdout-return
    Args:
        command_string (str): Command string that should be run outside of Python
    Returns:
        stdout (str): Standard-out of the command
        stderr (str): Potential error message caused by the command
        process (obj): An process object that can be used for further analysis
    """
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    return stdout, stderr, process
