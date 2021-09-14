import os
import pickle
import pytest
import numpy as np
import pandas as pd
from pqueens.main import main
from pqueens.utils import injector
from pqueens.tests.integration_tests.example_simulator_functions.park91a_hifi_coords import (
    park91a_hifi_coords,
)


def test_smc_park_hf(
    inputdir, tmpdir, design_and_write_experimental_data_to_csv, expected_samples, expected_weights
):
    """
    Integration test for bayesian multi-fidelity inverse analysis (bmfia)
    using the park91 function
    """

    # generate json input file from template
    template = os.path.join(inputdir, 'bmfia_smc_park.json')
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path, "plot_dir": tmpdir}
    input_file = os.path.join(tmpdir, 'smc_mf_park_realization.json')
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]

    # actual main call of smc
    main(arguments)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, 'smc_park_mf.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    samples = results['raw_output_data']['particles'].squeeze()
    weights = results['raw_output_data']['weights'].squeeze()

    # some tests / asserts here
    np.testing.assert_array_almost_equal(samples, expected_samples)
    np.testing.assert_array_almost_equal(weights.flatten(), expected_weights.flatten())


@pytest.fixture()
def expected_samples():
    samples = np.array(
        [
            [0.9387355, 0.59645344],
            [0.33947945, 0.59942216],
            [0.25040027, 0.5002987],
            [0.54835958, 0.25809571],
            [0.4199301, 0.50181386],
            [0.60370266, 0.83399187],
            [0.48832857, 0.30906785],
            [0.93761656, 0.6662429],
            [0.64660601, 0.06891406],
            [0.73610535, 0.26174618],
            [0.2701926, 0.94664253],
            [0.33468319, 0.26906282],
            [0.48068865, 0.27424412],
            [0.25393114, 0.59907956],
            [0.31428551, 0.72819402],
            [0.24361227, 0.96356603],
            [0.87378186, 0.5412163],
            [0.49589062, 0.84874959],
            [0.90643973, 0.8559544],
            [0.12630132, 0.04980633],
            [0.2987848, 0.55898757],
            [0.47339859, 0.46300629],
            [0.04813015, 0.14846379],
            [0.10618089, 0.37642431],
            [0.40414471, 0.8091166],
            [0.5845845, 0.3474076],
            [0.68922204, 0.59097884],
            [0.52428298, 0.92706009],
            [0.01881142, 0.93454701],
            [0.79131794, 0.93541572],
            [0.03833052, 0.47132136],
            [0.6486806, 0.44252171],
            [0.02333967, 0.33657389],
            [0.01465359, 0.35047296],
            [0.67727054, 0.61848785],
            [0.49305465, 0.68339117],
            [0.16371961, 0.46287103],
            [0.79577076, 0.24724528],
            [0.84537541, 0.38013469],
            [0.15661224, 0.38032687],
            [0.91406445, 0.48107137],
            [0.03185586, 0.22146798],
            [0.71735189, 0.23286237],
            [0.57145489, 0.42059946],
            [0.68414986, 0.77847095],
            [0.47504871, 0.53079667],
            [0.58477923, 0.85257618],
            [0.35647881, 0.37032978],
            [0.03562054, 0.54382245],
            [0.73921234, 0.86070219],
            [0.86615446, 0.66487258],
            [0.43660277, 0.8873714],
            [0.40951881, 0.02188231],
            [0.8962676, 0.48349995],
            [0.120641, 0.736617],
            [0.80650811, 0.42717846],
            [0.31608759, 0.51881708],
            [0.96079297, 0.09843122],
            [0.50465361, 0.79324524],
            [0.42183295, 0.10712982],
            [0.36474119, 0.81727009],
            [0.72777004, 0.55606159],
            [0.53052843, 0.43257183],
            [0.32400618, 0.81360703],
            [0.45870729, 0.57840285],
            [0.47883325, 0.0843024],
            [0.67003585, 0.17395736],
            [0.41538138, 0.32024194],
            [0.43528799, 0.93927203],
            [0.6152559, 0.13240175],
            [0.30961507, 0.41206015],
            [0.43345558, 0.67905015],
            [0.07142313, 0.14852126],
            [0.61366092, 0.19956105],
            [0.9285396, 0.81540356],
            [0.31194233, 0.28530911],
            [0.87696019, 0.62833916],
            [0.17249706, 0.22074853],
            [0.08755742, 0.65495369],
            [0.94089749, 0.12111906],
            [0.25937984, 0.63466202],
            [0.49211509, 0.82544313],
            [0.89107277, 0.53349398],
            [0.50265242, 0.87364668],
            [0.02892257, 0.4096777],
            [0.13538321, 0.08305636],
            [0.90081768, 0.74459972],
            [0.64406127, 0.32013523],
            [0.23345355, 0.19384469],
            [0.34612594, 0.41073842],
            [0.16400812, 0.63628919],
            [0.06488782, 0.15401756],
            [0.63308053, 0.94714138],
            [0.76968236, 0.37391638],
            [0.49947556, 0.40229961],
            [0.12840941, 0.80056603],
            [0.98328579, 0.09115912],
            [0.51992659, 0.70357057],
            [0.73629917, 0.40637812],
            [0.10489178, 0.12243632],
        ]
    )

    return samples


@pytest.fixture()
def expected_weights():
    weights = np.array(
        [
            [0.00618108],
            [0.00865758],
            [0.00514975],
            [0.01222551],
            [0.00690536],
            [0.01052084],
            [0.01200965],
            [0.00702389],
            [0.00693596],
            [0.0078329],
            [0.00569699],
            [0.00939414],
            [0.01067178],
            [0.01046555],
            [0.00728256],
            [0.02582272],
            [0.00892812],
            [0.0041041],
            [0.00906384],
            [0.00808595],
            [0.00748923],
            [0.01018432],
            [0.01090461],
            [0.00508103],
            [0.00773918],
            [0.00477677],
            [0.00740347],
            [0.0082557],
            [0.01421944],
            [0.00383957],
            [0.0056224],
            [0.0059007],
            [0.0274957],
            [0.01098109],
            [0.00698561],
            [0.00978043],
            [0.00983274],
            [0.01106626],
            [0.00827996],
            [0.00343605],
            [0.01100356],
            [0.00775984],
            [0.01019166],
            [0.00308316],
            [0.00708658],
            [0.00606985],
            [0.00738061],
            [0.0095975],
            [0.00251608],
            [0.00720447],
            [0.00756392],
            [0.00834777],
            [0.00671013],
            [0.00527688],
            [0.0066933],
            [0.0074906],
            [0.00967432],
            [0.00891426],
            [0.008031],
            [0.02046622],
            [0.00494212],
            [0.01522902],
            [0.04232537],
            [0.00687267],
            [0.00616423],
            [0.01051162],
            [0.00672912],
            [0.01412103],
            [0.01421131],
            [0.01654363],
            [0.00874611],
            [0.00939276],
            [0.00559782],
            [0.00586138],
            [0.00676958],
            [0.00678639],
            [0.01239949],
            [0.00427486],
            [0.05933396],
            [0.00329663],
            [0.00673911],
            [0.00784881],
            [0.06350215],
            [0.0060132],
            [0.00676228],
            [0.00593341],
            [0.00481306],
            [0.00875739],
            [0.00978495],
            [0.00621426],
            [0.00634032],
            [0.0191684],
            [0.00977437],
            [0.01254245],
            [0.0112311],
            [0.00480946],
            [0.00788359],
            [0.0065352],
            [0.00649602],
            [0.00544713],
        ]
    )
    return weights


@pytest.fixture()
def design_and_write_experimental_data_to_csv(tmpdir):
    # Fix random seed
    np.random.seed(seed=1)

    # create target inputs
    x1 = 0.6
    x2 = 0.4

    # use x3 and x4 as coordinates and create coordinate grid (same as in park91a_hifi_coords)
    xx3 = np.linspace(0.0, 1.0, 4)
    xx4 = np.linspace(0.0, 1.0, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # generate clean function output for fake test data
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(x1, x2, x3, x4))
    y_vec = np.array(y_vec)

    # add artificial noise to fake measurements
    sigma_n = 0.001
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        'x3': x3_vec,
        'x4': x4_vec,
        'y_obs': y_fake,
    }
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
