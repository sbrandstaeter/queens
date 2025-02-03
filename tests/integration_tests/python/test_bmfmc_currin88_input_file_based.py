#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Integration tests for the BMFMC routine.

The test is based on the high-fidelity Currin function and uses an input
file.
"""

# pylint: disable=invalid-name

import numpy as np
from scipy.stats import entropy

from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result
from queens.utils.pdf_estimation import estimate_pdf


# ---- actual integration tests -------------------------------------------------
def test_bmfmc_iterator_currin88_random_vars_diverse_design(
    tmp_path,
    inputdir,
    _write_lf_mc_data_to_pickle,
    hf_mc_data,
    bandwidth_lf_mc,
    design_method,
):
    """Integration tests for the BMFMC routine using an input file.

    The test uses a high-fidelity (HF) and a low-fidelity (LF) version
    of the *currin88* function.
    """
    # generate json input file from template
    template = inputdir / "bmfmc_currin88_template.yml"
    plot_dir = tmp_path
    lf_mc_data_name = "LF_MC_data.pickle"
    path_lf_mc_pickle_file = tmp_path / lf_mc_data_name
    dir_dict = {
        "lf_mc_pickle_file": path_lf_mc_pickle_file,
        "plot_dir": plot_dir,
        "design_method": design_method,
    }
    input_file = tmp_path / "bmfmc_currin88.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # get the results of the QUEENS run
    result_file = tmp_path / "bmfmc_currin88.pickle"
    results = load_result(result_file)

    # get the y_support and calculate HF MC reference
    y_pdf_support = results["raw_output_data"]["y_pdf_support"]

    p_yhf_mc, _ = estimate_pdf(
        np.atleast_2d(hf_mc_data).T, bandwidth_lf_mc, support_points=np.atleast_2d(y_pdf_support)
    )

    kl_divergence = entropy(p_yhf_mc, results["raw_output_data"]["p_yhf_mean"])
    assert kl_divergence < 0.3
