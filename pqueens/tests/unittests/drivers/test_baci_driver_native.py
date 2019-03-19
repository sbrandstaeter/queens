from pqueens.drivers.baci_driver_native import baci_driver_native

def test_gen_functionality_baci_driver_native(baci_job, baci_input_file, baci_output_file,
                           baci_cmd, baci_post_cmds, mocker):

    mock_inject = mocker.patch('pqueens.drivers.baci_driver_native.inject')
    mock_run_post_post_processing = mocker.patch('pqueens.drivers.baci_driver_native.run_post_post_processing')
    mock_run_post_processing = mocker.patch('pqueens.drivers.baci_driver_native.run_post_processing')
    mock_run_baci = mocker.patch('pqueens.drivers.baci_driver_native.run_baci')

    baci_driver_native(baci_job)

    mock_inject.assert_called_with(baci_job['params'],
                                   baci_job['driver_params']['input_template'],
                                   baci_input_file)

    mock_run_baci.assert_called_with(baci_cmd)

    calls_to_run_post_processing = [mocker.call(baci_post_cmd) for baci_post_cmd in baci_post_cmds]
    mock_run_post_processing.assert_has_calls(calls_to_run_post_processing, any_order=True)

    mock_run_post_post_processing.assert_called_with(baci_job['driver_params']['post_post_script'],
                                      baci_output_file)
