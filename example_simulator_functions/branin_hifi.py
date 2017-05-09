import numpy as np

def branin_hifi(x, y):

    result = (-1.275*x**2 / np.pi**2 + 5.0*x/np.pi + y - 6.0)**2 + \
             (10.0 - 5.0/(4.0*np.pi))*np.cos(x) + 10.0

    print("Result {}".format(result))

    return result

def main(job_id, params):
    return branin_hifi(params['x'], params['y'])
