import subprocess

def run(path_to_scalarvalues_csv):
    """ Get last line of a csv file and return 20th entry """

    # assemble full filename
    my_file = path_to_scalarvalues_csv+'0.scalarvalues.csv'

    # system call to get last line
    p1 = subprocess.Popen(["cat", my_file], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["tail", "-1"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    output, _ = p2.communicate()

    entries = output.decode().strip('\n').split(",")
    return float(entries[19])
