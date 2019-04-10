from pqueens.drivers.baci_driver_docker import baci_driver_docker

def test_gen_functionality_baci_driver_docker(baci_docker_job, baci_input_file, baci_output_file,
                           baci_cmd, baci_post_cmds, docker_volume_map, mocker):

    mock_inject = mocker.patch('pqueens.drivers.baci_driver_docker.inject')
    mock_run_post_post_processing = mocker.patch('pqueens.drivers.baci_driver_docker.run_post_post_processing')
    mock_run_post_processing = mocker.patch('pqueens.drivers.baci_driver_docker.run_post_processing')
    mock_run_baci = mocker.patch('pqueens.drivers.baci_driver_docker.run_baci')

    baci_driver_docker(baci_docker_job)

    mock_inject.assert_called_with(baci_docker_job['params'],
                                   baci_docker_job['driver_params']['input_template'],
                                   baci_input_file)
    mock_run_baci.assert_called_with(baci_docker_job['driver_params']['docker_container'],
                                     baci_cmd, docker_volume_map)

    calls_to_run_post_processing = [mocker.call(baci_docker_job['driver_params']['docker_container'], baci_post_cmd, docker_volume_map) for baci_post_cmd in baci_post_cmds]
    mock_run_post_processing.assert_has_calls(calls_to_run_post_processing, any_order=True)

    mock_run_post_post_processing.assert_called_with(baci_docker_job['driver_params']['post_post_script'],
                                      baci_output_file)

# currently it seems impossible to mock the Docker client
# https://github.com/docker/docker-py/issues/1854
# so for now we do not test
# run_baci, run_post_processing, and run_post_post_processing
