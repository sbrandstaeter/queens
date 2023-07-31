"""From config create utils."""

import logging

from pqueens.data_processor import VALID_TYPES as VALID_DATA_PROCESSOR_TYPES
from pqueens.distributions import VALID_TYPES as VALID_DISTRIBUTION_TYPES
from pqueens.drivers import VALID_TYPES as VALID_DRIVER_TYPES
from pqueens.external_geometry import from_config_create_external_geometry
from pqueens.interfaces import VALID_TYPES as VALID_INTERFACE_TYPES
from pqueens.interfaces.interface import Interface
from pqueens.iterators import VALID_TYPES as VALID_ITERATOR_TYPES
from pqueens.iterators.iterator import Iterator
from pqueens.models import VALID_TYPES as VALID_MODEL_TYPES
from pqueens.models.bmfmc_model import BMFMCModel
from pqueens.parameters.parameters import from_config_create_parameters
from pqueens.schedulers import VALID_TYPES as VALID_SCHEDULER_TYPES
from pqueens.schedulers.scheduler import Scheduler
from pqueens.utils.exceptions import InvalidOptionError
from pqueens.utils.experimental_data_reader import (
    VALID_TYPES as VALID_EXPERIMENTAL_DATA_READER_TYPES,
)
from pqueens.utils.import_utils import get_module_class
from pqueens.utils.iterative_averaging_utils import VALID_TYPES as VALID_ITERATIVE_AVERAGING_TYPES
from pqueens.utils.stochastic_optimizer import VALID_TYPES as VALID_STOCHASTIC_OPTIMIZER_TYPES

_logger = logging.getLogger(__name__)


VALID_TYPES = {
    **VALID_ITERATOR_TYPES,
    **VALID_MODEL_TYPES,
    **VALID_INTERFACE_TYPES,
    **VALID_DATA_PROCESSOR_TYPES,
    **VALID_STOCHASTIC_OPTIMIZER_TYPES,
    **VALID_DISTRIBUTION_TYPES,
    **VALID_EXPERIMENTAL_DATA_READER_TYPES,
    **VALID_ITERATIVE_AVERAGING_TYPES,
    **VALID_SCHEDULER_TYPES,
    **VALID_DRIVER_TYPES,
}


def from_config_create_iterator(config):
    """Create main iterator for queens run from config.

    A bottom up approach is used here to create all objects from the description. First, the objects
    that do not need any other so far uninitialized objects for initialization are initialized.
    These objects are put into the description of the other objects, where they are referenced.
    Then again the objects that do not need any other so far uninitialized objects for
    initialization are initialized. This process repeats until the main iterator (indicated by the
    'method' keyword) is initialized.

    Args:
        config (dict): Description of the queens run

    Returns:
        new_obj (iterator): Main queens iterator with all initialized objects.
    """
    # do pre-processing
    pre_processer = from_config_create_external_geometry(config, 'pre_processing')
    if pre_processer:
        config.pop('pre_processing')
        pre_processer.main_run()
        pre_processer.write_random_fields_to_dat()

    parameters = from_config_create_parameters(config.pop('parameters', {}), pre_processer)
    global_settings = config.pop('global_settings')
    obj_key = None
    while True:
        deadlock = True
        for obj_key, obj_dict in config.items():
            if isinstance(obj_dict, dict):
                reference_to_uninitialized_object = check_for_reference(obj_dict)
                if not reference_to_uninitialized_object:
                    deadlock = False
                    break
        if deadlock or obj_key is None:
            raise RuntimeError(
                "Queens run can not be configured due to missing 'method' "
                "description, circular dependencies or missing object descriptions! "
                f"Remaining uninitialized objects are: {list(config.keys())}"
            )

        try:
            if config[obj_key]['type'] in ["baci_dat"]:  # TODO: refactor fcc of external geometry
                new_obj = from_config_create_external_geometry(config, obj_key)
                config.pop(obj_key)
            else:
                obj_description = config.pop(obj_key)
                new_obj = from_config_create_object(obj_description, global_settings, parameters)
        except (TypeError, InvalidOptionError) as err:
            raise InvalidOptionError(f"Object '{obj_key}' can not be initialized.") from err

        if obj_key == 'method':
            if config:
                _logger.warning('Unused settings:')
                _logger.warning(config)
            return new_obj  # returns initialized iterator

        config = insert_new_obj(config, obj_key, new_obj)


def from_config_create_object(obj_description, global_settings=None, parameters=None):
    """Create object from description.

    Args:
        obj_description (dict): Description of the object
        global_settings (dict, optional): Global settings for the QUEENS run.
        parameters (obj, optional): Parameters object

    Returns:
        obj: Initialized object
    """
    object_class = get_module_class(obj_description, VALID_TYPES)
    if issubclass(object_class, (Iterator, Scheduler)):
        obj_description['global_settings'] = global_settings
    if issubclass(object_class, (Iterator, Interface, BMFMCModel)):
        obj_description['parameters'] = parameters
    return object_class(**obj_description)


def check_for_reference(obj_description):
    """Check if another uninitialized object is referenced.

    Indicated by a keyword that ends with '_name'. Sub-dictionaries are also checked.

    Args:
        obj_description (dict): Description of the object

    Returns:
        bool: True, if another uninitialized object is referenced.
    """
    for key, value in obj_description.items():
        if key.endswith('_name') and key != 'plot_name':  # TODO: rename plot_name keyword
            return True
        if isinstance(value, dict):
            reference_check = check_for_reference(value)
            if reference_check:
                return True
    return False


def insert_new_obj(config, new_obj_key, new_obj):
    """Insert initialized object in other object descriptions.

    Args:
        config (dict): Description of queens run, or sub dictionary
        new_obj_key (str): Key of initialized object
        new_obj (obj): Initialized object

    Returns:
        bool: True, if another uninitialized object is referenced.
    """
    referenced_keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = insert_new_obj(value, new_obj_key, new_obj)
        elif key.endswith('_name') and value == new_obj_key:
            referenced_keys.append(key)

    for key in referenced_keys:
        config.pop(key)  # remove key "<example>_name"
        config[key.removesuffix("_name")] = new_obj  # add key "<example>" with initialized object
    return config
