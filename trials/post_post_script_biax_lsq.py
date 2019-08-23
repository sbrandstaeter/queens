import numpy as np
import glob

def run(path_to_monitor_files):
    """ Compute objective function for optimization

        Args:
            path_to_monitor_files (string): path to monitor files to read
    """
    files = glob.glob(path_to_monitor_files+'*.mon')
    #print("found files {}".format(files))

    # get stem name get all files with that name
    disp_1320 = np.loadtxt(files[0], comments="#", skiprows=4, unpack=False, usecols=(2))
    disp_210 = np.loadtxt(files[1], comments="#", skiprows=4, unpack=False, usecols=(3))

    ref_disp_y_210 = np.array([3.261896e-03,
                               5.280317e-03,
                               9.039327e-03,
                               1.397350e-02,
                               2.032769e-02,
                               2.819794e-02,
                               3.585825e-02,
                               4.581416e-02,
                               6.044027e-02,
                               8.539806e-02,
                               1.093397e-01,
                               1.344145e-01,
                               1.650567e-01,
                               2.056554e-01,
                               2.515925e-01,
                               3.005144e-01,
                               3.755406e-01,
                               4.559404e-01,
                               5.587741e-01,
                               6.492275e-01,
                               7.758090e-01,
                               9.104572e-01,
                               1.068424e+00,
                               1.248679e+00,
                               1.456678e+00,
                               1.680452e+00,
                               1.949105e+00,
                               2.185310e+00,
                               2.465629e+00,
                               2.972625e+00,
                               3.351519e+00,
                               3.752029e+00,
                               4.149921e+00,
                               4.533758e+00,
                               4.936398e+00,
                               5.320126e+00,
                               5.689165e+00,
                               5.981061e+00,
                               6.315125e+00,
                               6.779584e+00])

    ref_disp_x_1320 = np.array([9.596285e-03,
                                2.509246e-02,
                                4.022012e-02,
                                5.713253e-02,
                                7.223338e-02,
                                9.432561e-02,
                                1.199008e-01,
                                1.452535e-01,
                                1.765155e-01,
                                2.277822e-01,
                                2.634707e-01,
                                3.077455e-01,
                                3.558735e-01,
                                4.101925e-01,
                                4.689172e-01,
                                5.333071e-01,
                                6.007081e-01,
                                6.815241e-01,
                                7.677312e-01,
                                8.502355e-01,
                                9.618579e-01,
                                1.066249e+00,
                                1.188211e+00,
                                1.312653e+00,
                                1.463981e+00,
                                1.617394e+00,
                                1.800095e+00,
                                1.952150e+00,
                                2.152836e+00,
                                2.477468e+00,
                                2.694511e+00,
                                2.915563e+00,
                                3.161252e+00,
                                3.414244e+00,
                                3.672826e+00,
                                3.888642e+00,
                                4.106300e+00,
                                4.297880e+00,
                                4.455837e+00,
                                4.743200e+00])


    # compute mean squared error
    residuals = np.append(ref_disp_y_210 - disp_210, ref_disp_x_1320 - disp_1320)
    return residuals

if __name__ == '__main__':
   run('/home/brandstaeter/coding/queens_dev/metropolis_hastings/output/OptimizeBiaxLSQ_1')
