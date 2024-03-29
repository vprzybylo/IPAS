"""
Calculates statistics on a group of IPAS particles
mean +/- one std
min and max of batch
mode of histogram
characteristic value of gamma distribution
"""

import statistics

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def auto_str(cls):
    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%s" % item for item in vars(self).items()),
        )

    cls.__str__ = __str__
    return cls


@auto_str
class Batch:
    def __init__(self, data):

        self.data = data[
            (data < np.quantile(data, 0.99)) & (data > np.quantile(data, 0.01))
        ]
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.pos_error = self.mean + self.std
        self.neg_error = self.mean - self.std
        self.min = min(self.data)
        self.max = max(self.data)
        # self.mode = statistics.mode(self.data)
        self.gamma_ch = None

    def mode_of_hist(self, plot=False):

        bins = (self.max - self.min) / 0.01
        values, bin_edges = np.histogram(self.data, bins=int(bins), density=True)
        if plot:
            values, bins, patches = plt.hist(
                self.data,
                bins=int(bins),
                density=True,
                color="navy",
                range=(self.min, self.max),
            )
            plt.show()
        self.mode = bin_edges[int(np.argwhere(values == np.max(values))[0])]

    def fit_distribution(self, plot=False, **kwargs):

        # Get histogram of original data
        values, bin_edges = np.histogram(self.data, density=True)
        fit_alpha, fit_loc, fit_beta = st.gamma.fit(self.data)
        pdf = st.gamma.pdf(bin_edges, a=fit_alpha, loc=fit_loc, scale=fit_beta)
        indmax = np.argmax(pdf)  # FIRST index where the highest prob occurs
        self.gamma_ch = bin_edges[indmax]  # characteristic of the distribution

        if plot:
            plt.hist(self.data, bins=70, density=True, color="navy", **kwargs)

            plt.plot(bin_edges, pdf, lw=5, color="darkorange")
            plt.ylim(0, max(values))
            plt.show()
