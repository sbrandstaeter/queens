from pqueens.drivers.ansys_driver import ANSYSDriver
from pqueens.drivers.driver import Driver
import pqueens


def test_init(mocker):
    base_settings = {'option': 'option_1'}

    mocker.patch('pqueens.drivers.driver.Driver.__init__')

    my_driver = ANSYSDriver(base_settings)

    pqueens.drivers.driver.Driver.__init__.assert_called_once_with(base_settings)


def test_from_config_create_driver(fake_database, mocker):
    mocker.patch('pqueens.drivers.ansys_driver.' 'ANSYSDriver.__init__', return_value=None)

    mocker.patch(
        'pqueens.database.mongodb.MongoDB.from_config_create_database', return_value=fake_database
    )

    base_settings = {'option': 'option_1'}
    base_settings['experiment_name'] = 'experiment_name'
    base_settings['experiment_dir'] = 'experiment_dir'
    base_settings['job_id'] = 666

    ANSYSDriver.from_config_create_driver(base_settings)
    pqueens.drivers.ansys_driver.ANSYSDriver.__init__.assert_called_once_with(base_settings)


def test_run_job(ansys_driver, mocker):
    mocker.patch.object(ansys_driver, 'assemble_ansys_run_cmd', return_value='stuff')
    m1 = mocker.patch(
        'pqueens.drivers.ansys_driver.run_subprocess',
        return_value=['random', 'stuff', 'out', 'err'],
    )

    ansys_driver.run_job()

    ansys_driver.assemble_ansys_run_cmd.assert_called_once()
    m1.assert_called_once_with('stuff', subprocess_type='simple')
