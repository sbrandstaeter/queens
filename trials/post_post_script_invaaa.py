import numpy as np

def run(path_to_monitor_file):
    line = np.loadtxt(path_to_monitor_file, comments="#", skiprows=4, unpack=False)
    result = np.sqrt(line[1]**2+line[2]**2+line[3]**2)
    return result
