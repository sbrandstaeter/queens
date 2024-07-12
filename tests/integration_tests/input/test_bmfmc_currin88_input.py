"""TODO_doc."""

# pylint: disable=invalid-name

import numpy as np
from scipy.stats import entropy

import queens.utils.pdf_estimation as est
from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result


# ---- actual integration tests -------------------------------------------------
def test_bmfmc_iterator_currin88_random_vars_diverse_design(
    tmp_path,
    inputdir,
    _write_LF_MC_data_to_pickle,
    generate_HF_MC_data,
    generate_LF_MC_data,
    design_method,
):
    """TODO_doc: add a one-line explanation.

    Integration tests for the BMFMC routine based on the HF and LF
    *currin88* function.
    """
    # generate json input file from template
    template = inputdir / 'bmfmc_currin88_template.yml'
    plot_dir = tmp_path
    lf_mc_data_name = 'LF_MC_data.pickle'
    path_lf_mc_pickle_file = tmp_path / lf_mc_data_name
    dir_dict = {
        'lf_mc_pickle_file': path_lf_mc_pickle_file,
        'plot_dir': plot_dir,
        'design_method': design_method,
    }
    input_file = tmp_path / 'bmfmc_currin88.yml'
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # get the results of the QUEENS run
    result_file = tmp_path / 'bmfmc_currin88.pickle'
    results = load_result(result_file)

    # get the y_support and calculate HF MC reference
    y_pdf_support = results['raw_output_data']['y_pdf_support']
    Y_LFs_mc = generate_LF_MC_data
    Y_HF_mc = generate_HF_MC_data
    bandwidth_lfmc = est.estimate_bandwidth_for_kde(
        Y_LFs_mc[:, 0], np.amin(Y_LFs_mc[:, 0]), np.amax(Y_LFs_mc[:, 0])
    )

    p_yhf_mc, _ = est.estimate_pdf(
        np.atleast_2d(Y_HF_mc).T, bandwidth_lfmc, support_points=np.atleast_2d(y_pdf_support)
    )

    kl_divergence = entropy(p_yhf_mc, results['raw_output_data']['p_yhf_mean'])
    assert kl_divergence < 0.3
