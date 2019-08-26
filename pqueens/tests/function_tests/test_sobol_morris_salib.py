# import os
# import pickle
#
# import pytest
#
# from pqueens.main import main
# TODO fix these test, because as of now these test produce platform dependent
# resutls
# def test_ishigami_morris_salib(inputdir, tmpdir):
#     """ Test case for salib based morris iterator """
#     arguments = ['--input=' + os.path.join(inputdir, 'sobol_morris_salib.json'),
#                  '--output='+str(tmpdir)]
#
#     main(arguments)
#     result_file = str(tmpdir)+'/'+'xxx.pickle'
#     with open(result_file, 'rb') as handle:
#         results = pickle.load(handle)
#
#     #print(results)
#
#     assert results["sensitivity_incides"]['mu'][0] == pytest.approx(-0.64045626)
#     assert results["sensitivity_incides"]['mu'][1] == pytest.approx(0.16710309)
#     assert results["sensitivity_incides"]['mu'][2] == pytest.approx(0.0433457)
#     assert results["sensitivity_incides"]['mu'][3] == pytest.approx(-0.43844718)
#     assert results["sensitivity_incides"]['mu'][4] == pytest.approx(0.03842642)
#     assert results["sensitivity_incides"]['mu'][5] == pytest.approx(0.0278994)
#     assert results["sensitivity_incides"]['mu'][6] == pytest.approx(0.01451847)
#     assert results["sensitivity_incides"]['mu'][7] == pytest.approx(-0.00757501)
#
#     assert results["sensitivity_incides"]['sigma'][0] == pytest.approx(7.17899944)
#     assert results["sensitivity_incides"]['sigma'][1] == pytest.approx(5.06120845)
#     assert results["sensitivity_incides"]['sigma'][2] == pytest.approx(1.99699929)
#     assert results["sensitivity_incides"]['sigma'][3] == pytest.approx(1.01166141)
#     assert results["sensitivity_incides"]['sigma'][4] == pytest.approx(0.09723708)
#     assert results["sensitivity_incides"]['sigma'][5] == pytest.approx(0.10568931)
#     assert results["sensitivity_incides"]['sigma'][6] == pytest.approx(0.10594465)
#     assert results["sensitivity_incides"]['sigma'][7] == pytest.approx(0.10770568)
