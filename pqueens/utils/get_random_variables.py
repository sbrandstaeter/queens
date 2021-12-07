from pqueens.utils.mcmc_utils import create_proposal_distribution


def get_random_variables(model):
    """Get random variables and fields from model.

    Args:
        model (model): instance of model class

    Returns:
        random_variables (dict): random variables
        random_fields (dict): random fields
        number_input_dimensions (int): number of input parameters/random variables
        distribution_info (list): information about distribution of random variables
    """
    # get random variables (RV) from model
    parameters = model.get_parameter()
    random_variables = parameters.get("random_variables", None)
    if random_variables is not None:
        number_input_dimensions = len(random_variables)
    else:
        raise RuntimeError("Random variables not correctly specified.")

    # get random fields (RF)
    random_fields = parameters.get("random_fields", None)

    distribution_info = get_distribution_info(random_variables)

    return random_variables, random_fields, number_input_dimensions, distribution_info


def get_distribution_info(random_variables):
    """Get distribution info.

    Args:
        random_variables (dict): random variables

    Return:
        distribution_info (list): information about distribution of random variables
    """
    distribution_info = []
    for _, rv in random_variables.items():
        temp = {
            "distribution": rv["distribution"],
            "distribution_parameter": rv["distribution_parameter"],
        }
        distribution_info.append(temp)
    return distribution_info


def get_random_samples(description, num_samples):
    """Get random samples based on QUEENS description of distribution.

    Args:
        description (dict):         Dictionary containing QUEENS distribution
                                    description
        num_samples (int):          Number of samples to generate

    Returns:
        np.array:                   Array with samples
    """

    distribution = create_proposal_distribution(description)
    samples = distribution.draw(num_draws=num_samples)

    return samples
