"""Random Fields."""

from pqueens.parameters.fields.random_fields import RandomField


def from_config_create_random_field(rf_dict, coords):
    """Create random fields from problem description.

    Args:
        rf_dict (dict):       Dictionary with random field description
        coords (dict): Dictionary with coordinates of discretized random field and the corresponding
                       keys

    Returns:
        rf: Random fields object
    """
    mean_param = rf_dict.get('mean_param')
    std_hyperparam_rf = rf_dict.get('std_hyperparam_rf')
    corr_length = rf_dict.get('corr_length')
    mean_type = rf_dict.get('mean_type')
    dimension = len(coords['keys'])
    explained_variance = rf_dict.get('explained_variance', 0.95)

    return RandomField(
        dimension=dimension,
        coords=coords,
        mean_param=mean_param,
        std_hyperparam_rf=std_hyperparam_rf,
        corr_length=corr_length,
        mean_type=mean_type,
        explained_variance=explained_variance,
    )
