"""Iterator to run a single simulation run."""

import numpy as np

from .iterator import Iterator


class SingleSimRunIterator(Iterator):
    """Iterator for single simulation run.

    Attributes:
        num_samples (int): Number of samples to compute.
        samples (np.array): Array with all samples.
        output (np.array): Array with all model outputs.
    """

    def __init__(self, model, global_settings):
        """Initialise Single Sim Run Iterator.

        Args:
            model (obj, optional): Model to be evaluated by iterator.
            global_settings (dict, optional): Settings for the QUEENS run.
        """
        super().__init__(model, global_settings)
        self.num_samples = 1
        self.samples = np.zeros(1)
        self.output = None

    def pre_run(self):
        """Generate samples for subsequent MC analysis and update model."""
        pass

    def core_run(self):
        """Run single simulation."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Not required here."""
        pass
