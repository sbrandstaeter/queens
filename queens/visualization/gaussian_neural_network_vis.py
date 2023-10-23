"""Plotting functions for the Gaussian Neural Network."""
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss(history, loss_plot_path):
    """Plot the loss function over the training epochs.

    Args:
        history (obj): Tensorflow history object of the training routine
        loss_plot_path (str): Path to save the loss plot
    """
    _, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.set_ylabel("-log lik.")
    ax.set_xlabel("# epochs")
    plt.savefig(Path(loss_plot_path, 'loss_plot.jpg'), dpi=300)
