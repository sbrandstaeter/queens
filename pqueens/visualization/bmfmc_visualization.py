import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

"""
A module that provides utilities and a class for visualization in BMFMC-UQ
It is designed such that the BMFMCVisualization class needs only to be initialized one
and can then be accessed and modified in the entire project.

In this context "this" is a pointer to the module object instance itself and can be compared to the 
"self" keyword in classes.

Attributes:
    bmfmc_visualization_instance (obj): Instance of the BMFMCVisualization class
    
Returns:
    None
"""

this = sys.modules[__name__]
this.bmfmc_visualization_instance = None


def from_config_create(config):
    """Module function that calls the class function `from_config_create` and
    creates instance of the BMFMCVisualization class from the problem
    description.

    Args:
        config (dict): Dictionary created from the input file, containing the problem description
    """
    this.bmfmc_visualization_instance = BMFMCVisualization.from_config_create(config)


class BMFMCVisualization(object):
    """Visualization class for BMFMC-UQ that containts several plotting,
    storing and visualization methods that can be used anywhere in QUEENS.

    Attributes:
       paths (list): List with paths to save the plots.
       save_bools (list): List with booleans to save plots.
       animation_bool (bool): Flag for animation of 3D plots.
       predictive_var (bool): Flag for predictive variance plots.
       no_features_ref (bool): Flag for BMFMC-reference without informative features plot.
       plot_booleans (list): List of booleans for determining whether individual plots should be
                             plotted or not.

    Returns:
        BMFMCVisualization (obj): Instance of the BMFMCVisualization Class
    """

    # some overall class states
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 28})

    def __init__(
        self, paths, save_bools, animation_bool, predictive_var, no_features_ref, plot_booleans
    ):
        self.paths = paths
        self.save_bools = save_bools
        self.animation_bool = animation_bool
        self.predictive_var = predictive_var
        self.no_features_ref = no_features_ref
        self.plot_booleans = plot_booleans

    @classmethod
    def from_config_create(cls, config):
        """
        Create the BMFMC visualization object from the problem description
        Args:
            config (dict): Dictionary containing the problem description

        Returns:
            Instance of BMFMCVisualization (obj)

        """

        method_options = config["method"].get("method_options")
        plotting_options = method_options["result_description"].get("plotting_options")
        paths = [
            os.path.join(plotting_options.get("plotting_dir"), name)
            for name in plotting_options["plot_names"]
        ]
        save_bools = plotting_options.get("save_bool")
        animation_bool = plotting_options.get("save_bool")
        predictive_var = method_options.get("predictive_var")
        no_features_ref = method_options.get("BMFMC_reference")
        plot_booleans = plotting_options.get("plot_booleans")
        return cls(
            paths, save_bools, animation_bool, predictive_var, no_features_ref, plot_booleans
        )

    def plot_pdfs(self, output):
        """Plot the output distributions of HF output prediction, LF output,
        Monte-Carlo reference, posterior variance or BMFMC reference with no
        features, depending on settings in input file. Animate and save plots
        dependent on problem description.

        Args:
            output (dict): dictionary containing key-value paris to plot

        Returns:
            Plots of model output distribution
        """
        if self.plot_booleans[0]:
            fig, ax = plt.subplots()

            min_x = min(output['y_pdf_support'])
            max_x = max(output['y_pdf_support'])
            min_y = 0
            max_y = 1.1 * max(output['p_yhf_mc'])
            ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))

            # --------------------- PLOT THE BMFMC POSTERIOR PDF MEAN ---------------------
            ax.plot(
                output['y_pdf_support'],
                output['p_yhf_mean'],
                color='xkcd:green',
                linewidth=3,
                label=r'$\mathrm{\mathbb{E}}_{f^*}\left[p\left(y^*_{\mathrm{HF}}|'
                r'f^*,\mathcal{D}_f\right)\right]$',
            )

            # ------------ plot the MC of first LF -------------------------------------------
            # Attention: we plot only the first p_ylf here, even if several LFs were used!
            ax.plot(
                output['y_pdf_support'],
                output['p_ylf_mc'],
                linewidth=1.5,
                color='r',
                alpha=0.8,
                label=r'$p\left(y_{\mathrm{LF}}\right)$',
            )

            # ------------------------ PLOT THE MC REFERENCE OF HF ------------------------
            ax.plot(
                output['y_pdf_support'],
                output['p_yhf_mc'],
                color='black',
                linestyle='-.',
                linewidth=3,
                alpha=1,
                label=r'$p\left(y_{\mathrm{HF}}\right),\ (\mathrm{MC-ref.})$',
            )

            # --------- Plot the posterior variance -----------------------------------------
            if self.predictive_var is True:
                _plot_pdf_var(output)

            # ---- plot the BMFMC reference without features
            if self.no_features_ref is True:
                _plot_pdf_no_features(output, posterior_variance=self.predictive_var)

            # ---- some further settings for the axes ---------------------------------------
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$p(y)$')
            ax.grid(which='major', linestyle='-')
            ax.grid(which='minor', linestyle='--', alpha=0.5)
            ax.minorticks_on()
            ax.legend(loc='upper right')
            fig.set_size_inches(15, 15)

            if self.save_bools[0] is not None:  # TODO: probably change that list to a dictionary
                _save_plot(self.save_bools[0], self.paths[0])

            plt.show()

    def plot_manifold(self, output, Y_LFs_mc, Y_HF_mc, Y_HF_train):
        """Plot the probabilistic manifold of high-fidelity, low-fidelity
        outputs and informative features of the input space, depending on the
        description in the input file. Also plot the probabilistic mapping
        along with its training points. Potentially animate and save these
        plots.

        Args:
            output (dict): dictionary containing key-value paris to plot
            Y_LFs_mc (np.array): Low-fidelity output Monte-Carlo samples
            Y_HF_mc (np.array):  High-fidelity output Monte-Carlo reference samples
            Y_HF_train (np.array): High-fidelity output training points

        Returns:
            Plots of the probabilistic manifold
        """
        if self.plot_booleans[1]:
            manifold_plotter = _get_manifold_plotter(output)
            if manifold_plotter is not None:
                manifold_plotter(output, Y_LFs_mc, Y_HF_mc, Y_HF_train)

            if self.animation_bool is True:
                _animate_3d(output, Y_HF_mc, self.paths[1])

            if self.save_bools[1] is not None:
                _save_plot(self.save_bools[1], self.paths[1])

            plt.show()

    def plot_feature_ranking(self, dim_counter, ranking, iteration):
        """
        Plot the score/ranking of possible candidates of informative features
        :math:`\\gamma_i`. Only the candidates with the highest score will be
        considered for :math:`z_{LF}`

        Args:
            dim_counter: Input dimension corresponding to the scores. In the
                         next iteration counts will change as one informative features
                         was already determined and the counter reset.
                         Note: In the first iteration dim_counter coincides with the input
                         dimensions. Afterwards, it only enumerates the remaining dimensions
                         of the input.
            ranking: Vector with scores/ranking of candidates :math:`\\gamma_i`
            iteration: Current iteration to find most informative input features

        Returns:
            Plots of the ranking/scores of candidates for informative features of the
            input :math:`\\gamma_i`

        """
        if self.plot_booleans[2]:
            fig, ax = plt.subplots()
            width = 0.25
            ax.bar(dim_counter + width, ranking[:, 0], width, label='ylf', color='g')
            ax.grid(which='major', linestyle='-')
            ax.grid(which='minor', linestyle='--', alpha=0.5)
            ax.minorticks_on()
            ax.set_xlabel('Feature')
            ax.set_ylabel(r'Projection $\mathbf{t}$')
            ax.set_xticks(dim_counter)
            plt.legend()
            fig.set_size_inches(15, 15)

            name_split = self.paths[2].split('.')
            path = name_split[0] + f'_{iteration}.' + name_split[1]

            if self.save_bools[2] is not None:
                _save_plot(self.save_bools[2], path)

            plt.show()


# ------ helper functions ----------------------------------------------------------
def _get_manifold_plotter(output):
    """Service method that returns the proper manifold plotting function
    depending on the output.

    Args:
        output (dict): Dictionary containing several output quantities for plotting

    Returns:
        Proper plotting function for either 2D or 3D plots depending on the output.
    """
    if output['Z_mc'].shape[1] < 2:
        return _2d_manifold
    elif output['Z_mc'].shape[1] == 2:
        return _3d_manifold
    else:
        return None


def _3d_manifold(output, Y_LFs_mc, Y_HF_mc, Y_HF_train):
    """
    Plot the data manifold in three dimensions.
    Args:
        output (dict): Dictionary containing output data to plot.
        Y_LFs_mc (np.array): Vector with LF output Monte-Carlo data.
        Y_HF_mc (np.array): Vector with reference HF output Monte-Carlo data.
        Y_HF_train (np.array): HF output training vector.

    Returns:
        3D-plot of output data manifold and one informative input feature :math:`\\gamma_1`

    """
    fig3 = plt.figure(figsize=(10, 10))
    ax3 = fig3.add_subplot(111, projection='3d')

    ax3.plot_trisurf(
        output['Z_mc'][:, 0],
        output['Z_mc'][:, 1],
        output['m_f_mc'][:, 0],
        shade=True,
        cmap='jet',
        alpha=0.50,
    )
    ax3.scatter(
        output['Z_mc'][:, 0, None],
        output['Z_mc'][:, 1, None],
        Y_HF_mc[:, None],
        s=4,
        alpha=0.7,
        c='k',
        linewidth=0.5,
        cmap='jet',
        label='$\mathcal{D}_{\mathrm{MC}}$, (Reference)',
    )

    ax3.scatter(
        output['Z_train'][:, 0, None],
        output['Z_train'][:, 1, None],
        Y_HF_train[:, None],
        marker='x',
        s=70,
        c='r',
        alpha=1,
        label='$\mathcal{D}$, (Training)',
    )

    ax3.set_xlabel(r'$\mathrm{y}_{\mathrm{LF}}$')
    ax3.set_ylabel(r'$\gamma$')
    ax3.set_zlabel(r'$\mathrm{y}_{\mathrm{HF}}$')

    minx = np.min(output['Z_mc'])
    maxx = np.max(output['Z_mc'])
    ax3.set_xlim3d(minx, maxx)
    ax3.set_ylim3d(minx, maxx)
    ax3.set_zlim3d(minx, maxx)

    ax3.set_xticks(np.arange(0, 0.5, step=0.5))
    ax3.set_yticks(np.arange(0, 0.5, step=0.5))
    ax3.set_zticks(np.arange(0, 0.5, step=0.5))
    ax3.legend()


def _2d_manifold(output, Y_LFs_mc, Y_HF_mc, Y_HF_train):
    """Plot 2D output data manifold (:math:`y_{HF}-y_{LF}`.

    Args:
        output (dict): Dictionary containing output data to plot.
        Y_LFs_mc (np.array): Vector with LF output Monte-Carlo data.
        Y_HF_mc (np.array): Vector with reference HF output Monte-Carlo data.
        Y_HF_train (np.array): HF output training vector.

    Returns:
        2D-plot of data manifold (:math:`y_{HF}-y_{LF}`-dependency.
    """
    fig2, ax2 = plt.subplots()
    ax2.plot(
        Y_LFs_mc[:, 0],
        Y_HF_mc,
        linestyle='',
        markersize=5,
        marker='.',
        color='grey',
        alpha=0.5,
        label=r'$\mathcal{D}_{\mathrm{ref}}='
        r'\{Y_{\mathrm{LF}}^*,Y_{\mathrm{HF}}^*\}$, (Reference)',
    )

    ax2.plot(
        np.sort(output['Z_mc'][:, 0]),
        output['m_f_mc'][np.argsort(output['Z_mc'][:, 0])],
        color='darkblue',
        linewidth=3,
        label=r'$\mathrm{m}_{\mathcal{D}_f}(y_{\mathrm{LF}})$, (Posterior mean)',
    )

    ax2.plot(
        np.sort(output['Z_mc'][:, 0]),
        np.add(output['m_f_mc'], np.sqrt(output['var_y_mc']))[np.argsort(output['Z_mc'][:, 0])],
        color='darkblue',
        linewidth=2,
        linestyle='--',
        label=r'$\mathrm{m}_{\mathcal{D}_f}(y_{\mathrm{LF}})\pm \sqrt{\mathrm{v}_'
        r'{\mathcal{D}_f}(y_{\mathrm{LF}})}$, (Confidence)',
    )

    ax2.plot(
        np.sort(output['Z_mc'][:, 0]),
        np.add(output['m_f_mc'], -np.sqrt(output['var_y_mc']))[np.argsort(output['Z_mc'][:, 0])],
        color='darkblue',
        linewidth=2,
        linestyle='--',
    )

    ax2.plot(
        output['Z_train'],
        Y_HF_train,
        linestyle='',
        marker='x',
        markersize=8,
        color='r',
        alpha=1,
        label=r'$\mathcal{D}_{f}=\{Y_{\mathrm{LF}},Y_{\mathrm{HF}}\}$, (Training)',
    )

    ax2.plot(
        Y_HF_mc,
        Y_HF_mc,
        linestyle='-',
        marker='',
        color='g',
        alpha=1,
        linewidth=3,
        label=r'$y_{\mathrm{HF}}=y_{\mathrm{LF}}$, (Identity)',
    )

    ax2.set_xlabel(r'$y_{\mathrm{LF}}$')
    ax2.set_ylabel(r'$y_{\mathrm{HF}}$')
    ax2.grid(which='major', linestyle='-')
    ax2.grid(which='minor', linestyle='--', alpha=0.5)
    ax2.minorticks_on()
    ax2.legend()
    fig2.set_size_inches(15, 15)


def _animate_3d(output, Y_HF_mc, save_path):
    """Animate 3D-data plots for better visual representation of the data.
    Potentially save animation as mp4.

    Args:
        output(dict): Dictionary containing output data to plot.
        Y_HF_mc (np.array): Vector with HF output Monte-Carlo reference data.
        save_path (str): Path (with name) for saving the animation.

    Returns:
        Animation of 3D plots.
    """

    def init():
        ax = plt.gca()
        ax.scatter(
            output['Z_mc'][:, 0, None],
            output['Z_mc'][:, 1, None],
            Y_HF_mc[:, None],
            s=3,
            c='darkgreen',
            alpha=0.6,
        )
        ax.set_xlabel(r'$y_{\mathrm{LF}}$')
        ax.set_ylabel(r'$\gamma$')
        ax.set_zlabel(r'$y_{\mathrm{HF}}$')
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
        return ()

    def animate(i):
        ax = plt.gca()
        ax.view_init(elev=10.0, azim=i)
        return ()

    fig = plt.gcf()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
    # Save
    save_path = save_path.split('.')[0] + '.mp4'
    anim.save(save_path, fps=30, dpi=300, extra_args=['-vcodec', 'libx264'])


def _plot_pdf_var(output, reference_str=''):
    """Plot the root of the posterior variance (=SD) of HF output density
    estimate :math:`\\mathbb{V}_{ f}\\left[p(y_{HF}^*|f,\\mathcal{D})\\right]`
    in form of credible intervals around the mean prediction.

    Args:
        output (dict): Dictionary containing output data to plot
        reference_str (str): String variable containing extra information to alter the
                             name of the path to distinguish posteriors with or without informative
                             features

    Returns:
        Plot of credible intervals for predicted HF output densities
    """
    ax = plt.gca()
    variance_base = 'p_yhf_mean' + reference_str
    variance_type = 'p_yhf_var' + reference_str
    ub = output[variance_base] + 2 * np.sqrt(output[variance_type])
    lb = output[variance_base] - 2 * np.sqrt(output[variance_type])
    ax.fill_between(
        output['y_pdf_support'],
        ub,
        lb,
        where=ub > lb,
        facecolor='lightgrey',
        alpha=0.5,
        interpolate=True,
        label=r'$\pm2\cdot\mathbb{SD}_{f^*}\left[p\left(y_{\mathrm{HF}}^*'
        r'|f^*,\mathcal{D}_f\right)\right]$',
    )


def _save_plot(save_bool, path):
    """Save the plot to specified path.

    Args:
        save_bool (bool): Flag to decide whether saving option is triggered.
        path (str): Path where to save the plot.

    Returns:
        Saved plot.
    """
    if save_bool is True:
        plt.savefig(path, dpi=300)


def _plot_pdf_no_features(output, posterior_variance=False):
    """
    Plot reference BMFMC prediction without using informative features
    :math:`\\gamma_i`.

    Args:
        output (dict): Dictionary containing output quantities to be plotted.
        posterior_variance (bool): Flag determining whether credible intervals should be plotted
                                   as well.

    Returns:
        Plot of BMFMC-prediction without using informative features of the input.

    """
    ax = plt.gca()

    # plot the bmfmc approx mean
    ax.plot(
        output['y_pdf_support'],
        output['p_yhf_mean_BMFMC'],
        color='xkcd:green',
        linewidth=1.5,
        linestyle='--',
        alpha=1,
        label=r'$\mathrm{\mathbb{E}}_{f^*}\left[p\left(y^*_{\mathrm{HF}}'
        r'|f^*,\mathcal{D}_f\right)\right],\ (\mathrm{no\ features})$',
    )

    # plot the bmfmc var
    if posterior_variance is True:
        _plot_pdf_var(output, reference_str='_BMFMC')
