
from .lhs_designer import LatinHyperCubeDesigner
from .monte_carlo_designer import MonteCarloDesigner

class DesignerFactory(object):
    """ Create new designer """

    def create_designer(designer_type,params,seed,num_samples):
        """ Create designer based on passed designer type

        Args:
            designer_type (string): what kind of designer to create

        Returns:
            designer: designer object
        """

        if designer_type == 'lhs':
            designer = LatinHyperCubeDesigner(params,seed,num_samples)
        elif designer_type == 'monte carlo':
            raise NotImplementedError
            #designer = MonteCarloDesigner(params,seed,num_samples)
        else:
            raise ValueError('Unkown type of designer')

        return designer

    factory = staticmethod(create_designer)
