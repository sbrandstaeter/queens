import plotly
import plotly.graph_objs as go
import numpy as np


def plot_pdf(pdf_estimate, support_points, bayes=False):
    """ Create pdf plot based on passed data

        Args:
            pdf_estimate   (dict):      Estimate of pdf at supporting points
            support_points (np.array):  Supporting points
            bayes (bool):               Do we want to plot confidence intervals
    """
    mean_pdf = go.Scatter(
        x=support_points,
        y=pdf_estimate["mean"],
        mode='markers-line',
        name='Mean'
    )

    data = [mean_pdf]

    if bayes is True:
        q5_pdf = go.Scatter(
            x=support_points,
            y=pdf_estimate["q5"],
            mode='markers-line',
            name='5% quantile'
        )
        data.append(q5_pdf)

        q95_pdf = go.Scatter(
            x=support_points,
            y=pdf_estimate["q95"],
            mode='markers-line',
            name='95% quantile'
        )
        data.append(q95_pdf)

        median_pdf = go.Scatter(
            x=support_points,
            y=pdf_estimate["median"],
            mode='markers-line',
            name='median'
        )
        data.append(median_pdf)

    layout = dict(title='Probability Density Function',
                  xaxis=dict(title='QOI'),
                  yaxis=dict(title='PDF'),
                 )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='PDF.html', auto_open=True)

def plot_cdf(cdf_estimate, support_points, bayes=False):
    """ Create cdf plot based on passed data

        Args:
            cdf_estimate   (dict):      Estimate of cdf at supporting points
            support_points (np.array):  Supporting points
            bayes (bool):               Do we want to plot confidence intervals
    """
    # Create a trace
    mean_cdf = go.Scatter(
        x=support_points,
        y=cdf_estimate["mean"],
        mode='markers-line',
        name='Mean'
    )
    data = [mean_cdf]

    if bayes is True:
        q5_cdf = go.Scatter(
            x=support_points,
            y=cdf_estimate["q5"],
            mode='markers-line',
            name='5% quantile'
        )
        data.append(q5_cdf)

        q95_cdf = go.Scatter(
            x=support_points,
            y=cdf_estimate["q95"],
            mode='markers-line',
            name='95% quantile'
        )
        data.append(q95_cdf)

        median_cdf = go.Scatter(
            x=support_points,
            y=cdf_estimate["median"],
            mode='markers-line',
            name='median'
        )
        data.append(median_cdf)

        # in case we want to plot posterior samples
        #  my_post_samples = cdf_estimate["post_samples"]
        # for i in range(my_post_samples.shape[0]):
        #     sample_cdf = go.Scatter(
        #         x=support_points,
        #         y=my_post_samples[i, :],
        #         mode='markers-line',
        #         name='sample'
        #     )
        #     data.append(sample_cdf)

    layout = dict(title='Cumulative Density Function',
                  xaxis=dict(title='QOI'),
                  yaxis=dict(title='CDF'),
                 )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='CDF.html', auto_open=True)

def plot_icdf(icdf_estimate, support_points, bayes=False):
    """ Create icdf plot based on passed data

        Args:
            icdf_estimate   (dict):      Estimate of icdf at supporting points
            support_points (np.array):   Supporting points
            bayes (bool):                Do we want to plot confidence intervals
    """
    raise NotImplementedError


def plot_failprob(failprob_estimate, failure_threshold, bayes=False):
    """ Create failure probability plot based on passed data

        Args:
            failprob_estimate   (dict):    Estimate of failprob_estimate for
                                           failure thresholds
            failure_threshold (np.array):  Failure thresholds
            bayes (bool):                  Do we want to plot confidence intervals
    """
    raise NotImplementedError
