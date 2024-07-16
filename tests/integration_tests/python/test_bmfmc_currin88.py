"""TODO_doc."""

# pylint: disable=invalid-name
import numpy as np
from scipy.stats import entropy

import queens.utils.pdf_estimation as est
from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.bmfmc_iterator import BMFMCIterator
from queens.main import run_iterator
from queens.models.bmfmc_model import BMFMCModel
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.gp_approximation_gpflow import GPFlowRegressionModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


# ---- actual integration tests -------------------------------------------------
def test_bmfmc_iterator_currin88_random_vars_diverse_design(
    tmp_path,
    _write_LF_MC_data_to_pickle,
    generate_HF_MC_data,
    generate_LF_MC_data,
    design_method,
    global_settings,
):
    """TODO_doc: add a one-line explanation.

    Integration tests for the BMFMC routine based on the HF and LF
    *currin88* function.
    """
    plot_dir = tmp_path
    lf_mc_data_name = "LF_MC_data.pickle"
    path_lf_mc_pickle_file = tmp_path / lf_mc_data_name
    # Parameters
    x1 = UniformDistribution(lower_bound=0.0, upper_bound=1.0)
    x2 = UniformDistribution(lower_bound=0.0, upper_bound=1.0)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    probabilistic_mapping = GPFlowRegressionModel(
        train_likelihood_variance=False,
        number_restarts=2,
        number_training_iterations=1000,
        dimension_lengthscales=2,
    )
    interface = DirectPythonInterface(function="currin88_hifi", parameters=parameters)
    hf_model = SimulationModel(interface=interface)
    model = BMFMCModel(
        predictive_var=False,
        BMFMC_reference=False,
        y_pdf_support_min=-0.5,
        y_pdf_support_max=15.0,
        path_to_lf_mc_data=(path_lf_mc_pickle_file,),
        path_to_hf_mc_reference_data=None,
        features_config="opt_features",
        num_features=1,
        probabilistic_mapping=probabilistic_mapping,
        hf_model=hf_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    iterator = BMFMCIterator(
        global_settings=global_settings,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_booleans": [False, False, False],
                "plotting_dir": plot_dir,
                "plot_names": ["pdfs.eps", "manifold.eps", "ranking.eps"],
                "save_bool": [False, False, False],
                "animation_bool": False,
            },
        },
        initial_design={
            "num_HF_eval": 100,
            "num_bins": 50,
            "method": design_method,
            "seed": 1,
            "master_LF": 0,
        },
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # get the y_support and calculate HF MC reference
    y_pdf_support = results["raw_output_data"]["y_pdf_support"]
    Y_LFs_mc = generate_LF_MC_data
    Y_HF_mc = generate_HF_MC_data
    bandwidth_lfmc = est.estimate_bandwidth_for_kde(
        Y_LFs_mc[:, 0], np.amin(Y_LFs_mc[:, 0]), np.amax(Y_LFs_mc[:, 0])
    )

    p_yhf_mc, _ = est.estimate_pdf(
        np.atleast_2d(Y_HF_mc).T, bandwidth_lfmc, support_points=np.atleast_2d(y_pdf_support)
    )

    kl_divergence = entropy(p_yhf_mc, results["raw_output_data"]["p_yhf_mean"])
    assert kl_divergence < 0.3
