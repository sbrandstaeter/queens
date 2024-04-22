"""TODO_doc."""

import pytest

from queens.main import run
from queens.utils.io_utils import load_result


@pytest.mark.max_time_for_test(20)
def test_branin_latin_hyper_cube(inputdir, tmp_path):
    """Test case for latin hyper cube iterator."""
    run(inputdir / 'latin_hyper_cube_branin.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')
    assert results["mean"] == pytest.approx(53.17279969296224)
    assert results["var"] == pytest.approx(2581.6502630157715)
