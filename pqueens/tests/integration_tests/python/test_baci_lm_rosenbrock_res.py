import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pqueens import run


def test_baci_lm_rosenbrock_res(inputdir, tmpdir):
    """Test case for Levenberg Marquardt iterator."""
    run(Path(Path(inputdir, 'baci_lm_rosenbrock_res.json')), Path(tmpdir))

    result_file = Path(tmpdir, 'OptimizeLM.csv')

    data = pd.read_csv(
        result_file,
        sep='\t',
    )

    params = data.get('params').tail(1)
    dfparams = params.str.extractall(r'([+-]?\d+\.\d*e?[+-]?\d*)')
    dfparams = dfparams.astype(float)
    numpyparams = dfparams.to_numpy()

    np.testing.assert_allclose(numpyparams, np.array([[+1.0], [+1.0]]), rtol=1.0e-5)

    assert Path(tmpdir, 'OptimizeLM.html').is_file()

    pass
