"""
this module is called in kde_distribution_plots.ipynb

find polynomial regressions for aggregate aspect ratio as a function of number of monomers and monomer aspect ratio
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from sympy import S, printing, symbols


def auto_str(cls):
    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%s" % item for item in vars(self).items()),
        )

    cls.__str__ = __str__
    return cls


@auto_str
class KDE:
    """
    Kernel density estimation (KDE) is a
    non-parametric method for estimating the
    probability density function of a given random variable
    """

    def __init__(self, kde):
        self.kde = kde

    def estimate_kde(self):
        """Evaluate the estimated pdf on a grid of points"""
        xgrid = np.arange(0.0, 1.01, 0.01)
        Xgrid, Ygrid = np.meshgrid(xgrid, xgrid)

        Z = self.kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        return Z.reshape(Xgrid.shape)

    def modes(self):
        """Find modes of kde distributions"""
        Z = self.estimate_kde()
        # modes
        mode_ba = (np.where(Z == np.max(Z))[0] / 100)[0]
        mode_ca = (np.where(Z == np.max(Z))[1] / 100)[0]
        return mode_ba, mode_ca


@auto_str
class Plot:
    """
    plot number of monomers vs aspect ratios
    """

    def __init__(self, axs, modes_ba, modes_ca, phios):
        self.axs = axs
        self.modes_ba = modes_ba
        self.modes_ca = modes_ca
        self.phios = phios
        self.interval = 2
        self.order = 4
        self.Ns = np.arange(2, 151, self.interval)

        self.colors = [
            "#85200e",
            "#c96400",
            "#f9af72",
            "#c3c5c9",
            "#77986d",
            "#376da6",
            "#4f3667",
        ]

    def fit(self, phi):
        self.coefs_ba = np.polyfit(
            self.Ns, self.modes_ba[phi, :: self.interval], self.order
        )
        self.coefs_ca = np.polyfit(
            self.Ns, self.modes_ca[phi, :: self.interval], self.order
        )

        self.fit_ba = np.poly1d(self.coefs_ba)
        self.fit_ca = np.poly1d(self.coefs_ca)

    def label(self, phi, ba=True):
        """
        label each regression line with polyfit eq
        never got the exponent formatting working (it looks like multiplication)
        """
        prec = 3
        y = "$\phi_{ca}$" if ba else "$\phi_{ba}$"
        coef = self.coefs_ba if ba else self.coefs_ca
        phi_eq = f"({self.phios[phi]})="
        eq = "".join(
            [
                "{:2.{prec}E}$n_m^{x}$".format(j, x=(len(coef) - i - 1), prec=prec)
                if j < 0
                else "+{:2.{prec}E}$n_m^{x}$".format(
                    j, x=(len(coef) - i - 1), prec=prec
                )
                for i, j in enumerate(coef)
            ]
        )
        return y + phi_eq + eq

    def plot_fit(self, linestyle, phi, linewidth=3):

        self.axs[0].plot(
            self.Ns,
            self.fit_ba(self.Ns),
            linestyle=linestyle,
            linewidth=linewidth,
            color=self.colors[phi],
            label=self.label(phi, ba=True),
        )

        self.axs[1].plot(
            self.Ns,
            self.fit_ca(self.Ns),
            linestyle=linestyle,
            linewidth=linewidth,
            color=self.colors[phi],
            # label=self.label(phi, ba=False),
        )

    def scatter(self, phi):
        self.axs[0].scatter(
            self.Ns,
            self.modes_ba[phi, :: self.interval],
            color=self.colors[phi],
            alpha=0.3,
        )

        self.axs[1].scatter(
            self.Ns,
            self.modes_ca[phi, :: self.interval],
            color=self.colors[phi],
            alpha=0.3,
        )

    def axis_layout(self):
        self.axs[0].set_xlim(0.0, 150)
        self.axs[0].set_ylim(0.0, 1.0)
        self.axs[1].set_ylim(0.0, 1.0)
        self.axs[0].set_ylabel("$\phi_{ca}$", fontsize=30)
        self.axs[1].set_ylabel("$\phi_{ba}$", fontsize=30)
        self.axs[0].set_xlabel("Number of monomers ($\it{n}_m$)")
        self.axs[1].set_xlabel("Number of monomers ($\it{n}_m$)")
        self.axs[0].yaxis.grid(True, which="major")
        self.axs[1].yaxis.grid(True, which="major")
        self.axs[0].yaxis.set_ticks(np.arange(0.0, 1.0, 0.10))
        self.axs[1].yaxis.set_ticks(np.arange(0.0, 1.0, 0.10))
        xlabels = np.arange(10, 150, 10)
        xlabels = np.insert(xlabels, 0, 2)
        xlabels = np.insert(xlabels, -1, 150)
        self.axs[1].xaxis.set_ticks(xlabels)

    def plot(self, phi, linestyle):

        self.plot_fit(linestyle, phi)
        # self.scatter(phi)

        x = 0.5
        y = -1.2

        self.axs[0].legend(
            bbox_to_anchor=(x, y),
            loc="lower center",
            ncol=2,
            title="RANDOM                                                                                                                                    QUASI-HORIZONTAL",
        )  # fmt: on/off

        x = 0.5
        y = -1.3
        plt.rcParams["legend.title_fontsize"] = 18
        plt.rcParams["legend.fontsize"] = 16
        #         self.axs[1].legend(
        #             bbox_to_anchor=(x, y),
        #             loc="lower center",
        #             ncol=2,
        # title="RANDOM                                                                                                                                    QUASI-HORIZONTAL",
        #        )  # fmt: on/off

        self.axis_layout()
