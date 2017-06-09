class SobolAnalyzer(object):
    """ Class for computing Sobol indicess

    Attributes:

    problem  (dict) :
        The problem definition
    calc_second_order (bool) :
        Calculate second-order sensitivities (default True)
    num_bootstrap_samples (int) :
        The number of bootstrap samples (default 100)
    confidence_level (float) :
        The confidence interval level (default 0.95)

    """

def __init__(self,problem,calc_second_order=True,num_bootstrap_samples=100,
             confidence_level=0.95):

    """
    Args:

    problem  (dict) :
        The problem definition
    calc_second_order (bool) :
        Calculate second-order sensitivities (default True)
    num_bootstrap_samples (int) :
        The number of bootstrap samples (default 100)
    confidence_level (float) :
        The confidence interval level (default 0.95)

    """
    self.calc_second_order = calc_second_order
    self.num_bootstrap = num_bootstrap
    self.confidence_level  = confidence_level
    print("hello world")

def analyze(self, Y):
    """ Compute sensitivity indices for given samples Y

    Args:

    Y (numpy.array) :
        NumPy array containing the model outputs)

    Returns (dict) :
        dictionary with sensitivity indices
