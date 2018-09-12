# from pqueens.main import main
# import pytest
# import pickle
# TODO fix these test, because as of now these test produce platform dependent
# resutls
# def test_ishigami_morris_salib(tmpdir):
#     """ Test case for salib based morris iterator """
#     arguments = ['--input=pqueens/tests/function_tests/input_files/ishigami_morris_salib.json',
#                  '--output='+str(tmpdir)]
#
#     main(arguments)
#     result_file = str(tmpdir)+'/'+'xxx.pickle'
#     with open(result_file, 'rb') as handle:
#         results = pickle.load(handle)
#
#     print(results)
#
#     assert results["sensitivity_incides"]['mu'][0] == pytest.approx(-2.41504207e+01)
#     assert results["sensitivity_incides"]['mu'][1] == pytest.approx(1.26225897e+01)
#     assert results["sensitivity_incides"]['mu'][2] == pytest.approx(-5.70363996e-12)
#
#     assert results["sensitivity_incides"]['mu_star'][0] == pytest.approx(2.41504207e+01)
#     assert results["sensitivity_incides"]['mu_star'][1] == pytest.approx(2.78976691e+01)
#     assert results["sensitivity_incides"]['mu_star'][2] == pytest.approx(5.70363996e-12)
#
#     assert results["sensitivity_incides"]['sigma'][0] == pytest.approx(1.65750124e+01)
#     assert results["sensitivity_incides"]['sigma'][1] == pytest.approx(2.92131768e+01)
#     assert results["sensitivity_incides"]['sigma'][2] == pytest.approx(2.88449896e-12)
#
#     assert results["sensitivity_incides"]['mu_star_conf'][0] == pytest.approx(13.26737536608863)
#     assert results["sensitivity_incides"]['mu_star_conf'][1] == pytest.approx(4.49548691231593)
#     assert results["sensitivity_incides"]['mu_star_conf'][2] == pytest.approx(2.4516069312374997e-12)
