from pqueens.drivers.ansys_driver_native import AnsysDriverNative
from pqueens.drivers.driver import Driver
import pqueens


def test_init(mocker):
    custom_executable = 'my_custom_anysy'
    ansys_version = 'v15'
    base_settings = {'option': 'option_1'}

    mocker.patch('pqueens.drivers.driver.Driver.__init__')

    my_driver = AnsysDriverNative(custom_executable, ansys_version, base_settings)

    pqueens.drivers.driver.Driver.__init__.assert_called_once_with(base_settings)

    assert my_driver.ansys_version == ansys_version
    assert my_driver.custom_executable == custom_executable


def test_from_config_create_driver(mocker):
    mocker.patch(
        'pqueens.drivers.ansys_driver_native.' 'AnsysDriverNative.__init__', return_value=None
    )

    base_settings = {'option': 'option_1'}
    config = {'driver': {}}
    config['driver']['driver_params'] = {
        'custom_executable': 'my_custom_anysy',
        'ansys_version': 'v15',
    }

    AnsysDriverNative.from_config_create_driver(config, base_settings)
    pqueens.drivers.ansys_driver_native.AnsysDriverNative.__init__.assert_called_once_with(
        'my_custom_anysy', 'v15', base_settings
    )


def test_run_job(ansys_driver, mocker):
    mocker.patch.object(ansys_driver, 'assemble_command_string', return_value='stuff')
    m1 = mocker.patch(
        'pqueens.drivers.ansys_driver_native.run_subprocess',
        return_value=['random', 'stuff', 'out', 'err'],
    )

    ansys_driver.run_job()

    ansys_driver.assemble_command_string.assert_called_once()
    m1.assert_called_once_with('stuff')
