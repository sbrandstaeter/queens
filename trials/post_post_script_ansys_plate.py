import subprocess

def run(my_file):
    """ Get line of ansys .out file  return 1st entry """

    # system call to get last line
    p1 = subprocess.Popen(["cat", my_file], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["tail", "-1"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    output, _ = p2.communicate()

    entries = output.decode().strip('\n').split(" ")
    return float(entries[0])
