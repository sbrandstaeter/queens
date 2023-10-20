"""TODO_doc."""

import numpy as np
import pandas as pd

from queens.main import run


def test_baci_lm_rosenbrock_res(inputdir, tmp_path):
    """Test case for Levenberg Marquardt iterator."""
    run(inputdir / 'baci_lm_rosenbrock_res.yml', tmp_path)

    result_file = tmp_path / 'OptimizeLM.csv'

    data = pd.read_csv(
        result_file,
        sep='\t',
    )

    params = data.get('params').tail(1)
    dfparams = params.str.extractall(r'([+-]?\d+\.\d*e?[+-]?\d*)')
    dfparams = dfparams.astype(float)
    numpyparams = dfparams.to_numpy()

    np.testing.assert_allclose(numpyparams, np.array([[+1.0], [+1.0]]), rtol=1.0e-5)

    assert (tmp_path / 'OptimizeLM.html').is_file()
