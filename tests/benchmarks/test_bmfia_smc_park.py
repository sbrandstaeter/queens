"""Benchmark test for BMFIA with Python test function."""

import numpy as np
import pandas as pd
import pytest

from queens.example_simulator_functions.park91a import park91a_hifi
from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result
from test_utils.benchmarks import assert_weights_and_samples


def test_bmfia_park_hf_smc(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_weights,
    expected_samples,
):
    """Benchmark test for bayesian multi-fidelity inverse analysis (bmfia).

    In this test the park91 function is used instead of a simulation
    code.
    """
    # generate yaml input file from template
    template = inputdir / 'bmfia_smc_park.yml'
    experimental_data_path = tmp_path
    dir_dict = {'experimental_data_path': experimental_data_path, 'plot_dir': tmp_path}
    input_file = tmp_path / 'smc_mf_park_realization.yml'
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)

    assert_weights_and_samples(results, expected_weights, expected_samples)


@pytest.fixture(name="_create_experimental_data_park91a_hifi_on_grid")
def fixture_create_experimental_data_park91a_hifi_on_grid(tmp_path):
    """Fixture to write dummy observation data."""
    # Fix random seed
    np.random.seed(seed=1)

    # create target inputs
    x1 = 0.5
    x2 = 0.2

    # use x3 and x4 as coordinates and create coordinate grid (same as in park91a_hifi_coords)
    xx3 = np.linspace(0.0, 1.0, 4)
    xx4 = np.linspace(0.0, 1.0, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # generate clean function output for fake test data
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi(x1, x2, x3, x4))
    y_vec = np.array(y_vec)

    # add artificial noise to fake measurements
    sigma_n = 0.1
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        'x3': x3_vec,
        'x4': x4_vec,
        'y_obs': y_fake,
    }
    experimental_data_path = tmp_path / 'experimental_data.csv'
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="expected_weights")
def fixture_expected_weights():
    """Expected weights."""
    weights = np.array(
        [
            0.02026016,
            0.02060205,
            0.023068,
            0.02118768,
            0.00889649,
            0.0081705,
            0.00525636,
            0.00442393,
            0.04115287,
            0.00502698,
            0.01516821,
            0.01385408,
            0.00845317,
            0.00900209,
            0.00183214,
            0.00917747,
            0.02454498,
            0.00515997,
            0.00933705,
            0.00567719,
            0.00764971,
            0.01078292,
            0.00761101,
            0.00687728,
            0.00314144,
            0.01150485,
            0.0018492,
            0.01849746,
            0.00317111,
            0.00990198,
            0.00836321,
            0.00864723,
            0.00342693,
            0.00455993,
            0.01290228,
            0.00896136,
            0.00680431,
            0.01586553,
            0.00532754,
            0.00972338,
            0.00590847,
            0.01404336,
            0.03114785,
            0.02325,
            0.00773386,
            0.00693315,
            0.0060638,
            0.00622674,
            0.01262532,
            0.00952854,
            0.00644537,
            0.01477876,
            0.00572131,
            0.00294183,
            0.01320269,
            0.02456792,
            0.01485442,
            0.02893771,
            0.01283193,
            0.00638253,
            0.00533457,
            0.01851759,
            0.01035809,
            0.00354573,
            0.00407061,
            0.00238959,
            0.01540217,
            0.00530767,
            0.01073429,
            0.00281793,
            0.00836412,
            0.01502365,
            0.00818799,
            0.00196828,
            0.00831479,
            0.00704763,
            0.00818811,
            0.00175943,
            0.0002911,
            0.0089211,
            0.01502431,
            0.00751416,
            0.00379418,
            0.02956547,
            0.00475145,
            0.00655539,
            0.00974881,
            0.00486147,
            0.00278865,
            0.00625222,
            0.00858278,
            0.01739825,
            0.0043038,
            0.00642224,
            0.00465958,
            0.01588978,
            0.00333407,
            0.00716114,
            0.00276105,
            0.01214118,
        ]
    )
    return weights


@pytest.fixture(name="expected_samples")
def fixture_expected_samples():
    """Expected samples."""
    samples = np.array(
        [
            [0.4760531, 0.42023761],
            [0.51856308, 0.36197203],
            [0.51392015, -0.17227154],
            [0.48555291, 0.44354172],
            [0.51345517, 0.16328839],
            [0.50496001, 0.13204256],
            [0.53360589, 0.12008268],
            [0.50472891, 0.33446679],
            [0.54524547, -0.30743248],
            [0.51710612, 0.0246309],
            [0.47926825, 0.33064799],
            [0.44028432, 0.48096685],
            [0.53633648, -0.30374654],
            [0.53349927, -0.06847852],
            [0.51596435, -0.10227327],
            [0.51793232, 0.27542282],
            [0.51585513, 0.25843277],
            [0.53554383, 0.2735094],
            [0.52957643, 0.03224593],
            [0.48367072, 0.20354831],
            [0.52457692, -0.34956723],
            [0.53452122, 0.06103901],
            [0.42266607, 0.56684091],
            [0.5253056, 0.17640377],
            [0.54915851, 0.02795585],
            [0.56528344, -0.27530543],
            [0.46265082, 0.51043895],
            [0.44341381, 0.4849019],
            [0.49658479, 0.22935896],
            [0.48659612, 0.26547227],
            [0.54276782, -0.437803],
            [0.52489363, -0.10359931],
            [0.50396365, 0.23435098],
            [0.50824623, 0.20327055],
            [0.55291951, 0.15940135],
            [0.46984121, 0.44136251],
            [0.48996779, 0.28131321],
            [0.52147607, -0.0694735],
            [0.54670714, 0.16719876],
            [0.48100374, 0.23022417],
            [0.54311241, -0.16946221],
            [0.55061186, 0.12971585],
            [0.49193237, 0.42880887],
            [0.50078465, 0.31362242],
            [0.49131368, 0.30020734],
            [0.52457175, -0.1384063],
            [0.45490437, 0.41321264],
            [0.50957222, 0.09679816],
            [0.44721411, 0.43907272],
            [0.49090669, 0.52029561],
            [0.54846229, 0.06387224],
            [0.54308239, 0.25224447],
            [0.47500715, 0.32833396],
            [0.52604554, 0.1462472],
            [0.44530931, 0.45519931],
            [0.4735073, 0.41998891],
            [0.54959953, -0.28977119],
            [0.42507539, 0.57395492],
            [0.54766715, 0.27676876],
            [0.51470175, -0.23899977],
            [0.53790512, 0.30551773],
            [0.50918631, 0.18354487],
            [0.51651983, -0.00321281],
            [0.48971066, 0.33727104],
            [0.46187971, 0.52961428],
            [0.4755353, 0.26404212],
            [0.51502947, 0.17951405],
            [0.55860225, 0.14414969],
            [0.51731226, 0.15008172],
            [0.53091356, -0.08386808],
            [0.44077152, 0.3725247],
            [0.35686151, 0.72947217],
            [0.49266971, 0.29129961],
            [0.48473595, 0.38494753],
            [0.48040451, 0.20957119],
            [0.56130765, 0.14279768],
            [0.51521485, 0.22960488],
            [0.4759566, 0.33802613],
            [0.46352056, 0.41572131],
            [0.52705113, 0.27003072],
            [0.52234449, -0.32234947],
            [0.50131921, 0.35136083],
            [0.50747033, 0.40654347],
            [0.57760796, -0.11598008],
            [0.53299722, 0.37990487],
            [0.49352754, 0.38430099],
            [0.52752799, -0.32143072],
            [0.57281223, -0.03900162],
            [0.55027051, -0.24588637],
            [0.51823131, -0.02887667],
            [0.47121463, 0.40420202],
            [0.55219525, 0.07631147],
            [0.50963365, -0.18745337],
            [0.47156058, -0.06732817],
            [0.55015114, 0.15066165],
            [0.49785938, 0.4076165],
            [0.42163649, 0.39760788],
            [0.4446676, 0.58640318],
            [0.51805501, 0.27292856],
            [0.46923552, 0.41215362],
        ]
    )
    return samples
