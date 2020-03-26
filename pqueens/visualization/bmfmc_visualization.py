import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot_init():
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 28})


def plot_pdf_var(output, reference_str=''):
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


def save_plot(saving_config):
    if saving_config['saving_bool'] is True:
        plt.savefig(saving_config['path'], dpi=300)


def plot_pdf_no_features(output, posterior_variance=False):
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
        plot_pdf_var(output, reference_str='_BMFMC')


def plot_pdfs(output, posterior_variance=False, no_features_ref=False, saving_config=None):

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
    if posterior_variance is True:
        plot_pdf_var(output)

    # ---- plot the BMFMC reference without features
    if no_features_ref is True:
        plot_pdf_no_features(output, posterior_variance=posterior_variance)

    # ---- some further settings for the axes ---------------------------------------
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$p(y)$')
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--', alpha=0.5)
    ax.minorticks_on()
    ax.legend(loc='upper right')
    fig.set_size_inches(15, 15)

    if saving_config['saving_bool'] is not None:
        save_plot(saving_config)

    plt.show()


def plot_manifold(output, Y_LFs_mc, Y_HF_mc, Y_HF_train, saving_config=None, animation=False):

    if output['Z_mc'].shape[1] < 2:
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
            np.add(output['m_f_mc'], np.sqrt(output['var_f_mc']))[np.argsort(output['Z_mc'][:, 0])],
            color='darkblue',
            linewidth=2,
            linestyle='--',
            label=r'$\mathrm{m}_{\mathcal{D}_f}(y_{\mathrm{LF}})\pm \sqrt{\mathrm{v}_'
            r'{\mathcal{D}_f}(y_{\mathrm{LF}})}$, (Confidence)',
        )

        ax2.plot(
            np.sort(output['Z_mc'][:, 0]),
            np.add(output['m_f_mc'], -np.sqrt(output['var_f_mc']))[
                np.argsort(output['Z_mc'][:, 0])
            ],
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

    if output['Z_mc'].shape[1] == 2:
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

        if animation is True:
            animate_3d(output, Y_HF_mc, saving_config['path'])

    if saving_config['saving_bool'] is not None:
        save_plot(saving_config)

    plt.show()


def animate_3d(output, Y_HF_mc, save_path):
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
