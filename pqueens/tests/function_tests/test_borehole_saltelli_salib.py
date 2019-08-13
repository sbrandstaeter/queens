import os
import pickle

import pytest

from pqueens.main import main

def test_borehole_saltelli(inputdir, tmpdir):
    """ Test case for SA_lib saltelli iterator """
    arguments = ['--input=' + os.path.join(inputdir, 'borehole_salib_saltelli.json'),
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    #print("results {}".format(results))
    #print("Parameter Names {}".format(results["parameter_names"]))
    #print("sensitivity_incides][S1] {}".format(results["sensitivity_incides"]["S1"]))

    assert results["sensitivity_incides"]["S1"][0] == pytest.approx(8.24597209e-01)
    assert results["sensitivity_incides"]["S1"][1] == pytest.approx(6.09543158e-05)
    assert results["sensitivity_incides"]["S1"][2] == pytest.approx(1.48009577e-08)
    assert results["sensitivity_incides"]["S1"][3] == pytest.approx(4.04630186e-02)
    assert results["sensitivity_incides"]["S1"][4] == pytest.approx(-3.59511283e-05)
    assert results["sensitivity_incides"]["S1"][5] == pytest.approx(4.07853843e-02)
    assert results["sensitivity_incides"]["S1"][6] == pytest.approx(4.02116044e-02)
    assert results["sensitivity_incides"]["S1"][7] == pytest.approx(8.59606495e-03)
