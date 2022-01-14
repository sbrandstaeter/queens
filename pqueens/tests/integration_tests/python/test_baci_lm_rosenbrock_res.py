import os

import numpy as np
import pandas as pd
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_baci_lm_rosenbrock_res(inputdir, tmpdir):
    """Test case for Levenberg Marquardt iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'baci_lm_rosenbrock_res.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = os.path.join(tmpdir, 'OptimizeLM.csv')

    data = pd.read_csv(
        result_file,
        sep='\t',
    )

    params = data.get('params').tail(1)
    dfparams = params.str.extractall(r'([+-]?\d+\.\d*e?[+-]?\d*)')
    dfparams = dfparams.astype(float)
    numpyparams = dfparams.to_numpy()

    np.testing.assert_allclose(numpyparams, np.array([[+1.0], [+1.0]]), rtol=1.0e-5)

    assert os.path.isfile(os.path.join(tmpdir, 'OptimizeLM.html'))

    pass
