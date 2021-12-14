"""
holds plotting modules for area ratio, aspect ratio, and complexity
from CPI data with respect to campaign
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joypy import joyplot
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


def auto_str(cls):
    """
    string representation of all attributes of the class and the class name
    called with str(instance_name)
    """

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%s" % item for item in vars(self).items()),
        )

    cls.__str__ = __str__
    return cls


@auto_str
class Plot:
    def __init__(self, df_CPI):
        self.df_CPI = df_CPI
        self.particle_types = list(self.df_CPI["Classification"].unique())

    def plot_part_type(self, df_type, campaign, i, var, part_type, axs, c, colors_cpi):
        sns.kdeplot(
            df_type, color=colors_cpi[c], label=campaign, ax=axs[i], linewidth=3
        )
        no_ylab = [1, 2, 3, 5, 6, 7]
        axs[i].set_ylabel(" " if c in no_ylab else "Density")
        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[i].set_xlim(0.0, 1.0)

    def part_type_legend(self, axs, i):
        """legend"""
        y = -0.5
        x = -1.9
        axs[i].legend(
            bbox_to_anchor=(x, y),
            loc="lower center",
            ncol=7,
            title="CAMPAIGN",
            fontsize=14,
        )

    def change_title_name(self, part_type):
        part_type = "compact irregular" if part_type == "compact_irreg" else part_type
        part_type = "aggregate" if part_type == "agg" else part_type
        part_type = (
            "planar polycrystal" if part_type == "planar_polycrystal" else part_type
        )
        return part_type

    def colors(self):
        """cpi hex colors for campaigns"""
        cmap = cm.get_cmap("coolwarm", len(self.df_CPI["Campaign"].unique()))
        colors_cpi = []
        for i in range(cmap.N):
            rgba = cmap(i)
            # rgb2hex accepts rgb or rgba
            colors_cpi.append(matplotlib.colors.rgb2hex(rgba))
        return colors_cpi

    def plot_Dmax_phi(self, part_type="agg"):

        for part_type in self.particle_types:
            df = self.df_CPI[self.df_CPI["Classification"] == part_type]
            plt.figure(figsize=(5, 5))
            sns.kdeplot(data=df, x="Aspect Ratio", y="Dmax")

    def part_type_subplots(self, var="Aspect Ratio"):
        """
        plot cpi data for a specified variable with each subplot representing a particle type
        campaigns are plotted in different lines within each subplot
        """
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)
        axs = axs.ravel()
        particle_types = [
            "agg",
            "bullet",
            "column",
            "planar_polycrystal",
            "compact_irreg",
        ]

        colors_cpi = self.colors()
        # colors_cpi = ['k','#001219', '#005F73', '#0A9396','#94D2BD','#E9D8A6','#D4EDE5','gray','#EE9B00','#FFF1D6','#CA6702','#BB3E03','#AE2012','#962224','#430F10']
        lens = np.zeros(len(self.df_CPI["Campaign"].unique()))
        for c, campaign in enumerate(self.df_CPI["Campaign"].unique()):
            df = self.df_CPI[self.df_CPI["Campaign"] == campaign]
            for i, part_type in enumerate(particle_types):

                df_type = df[var][df["Classification"] == part_type]
                lens[c] += len(df_type)
                self.plot_part_type(
                    df_type, campaign, i, var, part_type, axs, c, colors_cpi
                )

        for i, part_type in enumerate(particle_types):
            part_type = self.change_title_name(part_type)
            axs[i].set_title(f"{part_type}\n n={lens[i]:,.0f}")
        self.part_type_legend(axs, i)
        plt.rcParams["legend.title_fontsize"] = "xx-large"
        plt.savefig(f"../plots/campaign_{var}.png", bbox_inches="tight")

    def part_type_ridgeplots(self, var="Aspect Ratio"):
        """
        plot cpi data for a specified variable with each subplot representing a particle type
        campaigns are plotted in different lines within each subplot
        """
        particle_types = [
            "agg",
            "bullet",
            "column",
            "planar_polycrystal",
            "compact_irreg",
        ]

        lens = np.zeros(len(self.df_CPI["Campaign"].unique()))
        for c, campaign in enumerate(self.df_CPI["Campaign"].unique()):
            df = self.df_CPI[self.df_CPI["Campaign"] == campaign]
            for i, part_type in enumerate(particle_types):
                df_type = df[var][df["Classification"] == part_type]
                lens[c] += len(df_type)

        for i, part_type in enumerate(particle_types):
            plt.figure(figsize=(5, 5))

            data = self.df_CPI[["Campaign", var]][
                self.df_CPI["Classification"] == part_type
            ]
            part_type = self.change_title_name(part_type)
            joyplot(
                title=f"{part_type}\n n={lens[i]:,.0f}",
                data=data,
                by="Campaign",
                colormap=plt.cm.jet,
                x_range=[0.0, 1.0],
            )

            plt.xlabel(f"{var}")
            plt.savefig(
                f"../plots/campaign_ridgeplot_{var}_{part_type}.png",
                dpi=300,
                bbox_inches="tight",
            )

    def plot_campaigns(self, df, campaign, i, var, part_type, axs, c, colors_cpi):
        df_type = df[var][df["Classification"] == part_type]
        part_type = "compact irregular" if part_type == "compact_irreg" else part_type
        part_type = (
            "planar polycrystal" if part_type == "planar_polycrystal" else part_type
        )

        sns.kdeplot(
            df_type, color=colors_cpi[i], label=part_type, ax=axs[c], linewidth=3
        )
        no_ylab = [1, 2, 3, 5, 6, 7]
        axs[c].set_ylabel(" " if c in no_ylab else "Density")
        axs[c].set_title(f"{campaign}\n n={len(df_type):,}")
        axs[c].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[c].set_xlim(0.0, 1.0)

    def campaign_legend(self, axs, c):
        y = -0.5
        x = -3.3
        axs[c].legend(
            bbox_to_anchor=(x, y),
            loc="lower center",
            ncol=5,
            title="Particle Classification",
            fontsize=14,
        )

    def campaign_subplots(self, var="Aspect Ratio"):
        colors_cpi = ["#890000", "#ae3d00", "#ba9600", "#3b5828", "#014c63"]
        particle_types = [
            "agg",
            "bullet",
            "column",
            "planar_polycrystal",
            "compact_irreg",
        ]
        fig, axs = plt.subplots(2, 7, figsize=(25, 10), sharex=True, sharey=True)
        axs = axs.ravel()

        for c, campaign in enumerate(self.df_CPI["Campaign"].unique()):
            self.df_CPI["Aspect Ratio"] = self.df_CPI["c"] / self.df_CPI["a"]
            df = self.df_CPI[self.df_CPI["Campaign"] == campaign]
            for i, part_type in enumerate(particle_types):
                self.plot_campaigns(df, campaign, i, var, part_type, axs, c, colors_cpi)

        self.campaign_legend(axs, c)
        plt.rcParams["legend.title_fontsize"] = "xx-large"
        plt.savefig(f"../plots/campaign_{var}.png", bbox_inches="tight")

    def boxplot(self, df, var, ax):
        sns.boxplot(
            x="binned",
            y=var,
            data=df,
            fliersize=0,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            ax=ax,
        )

    def complexity_area_ratio(self, part_type="agg"):
        """
        complexity (assuming increasing number of monomers with increase in C)
        vs. area ratio and aspect ratio in subplot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10), sharex=True)

        # truncate CPI dataframe for a specific particle type
        df = self.df_CPI[self.df_CPI["Classification"] == part_type].copy()

        # bin complexity into 10 linearly spaced bins
        df["binned"] = pd.cut(df["Complexity"], bins=np.linspace(0.0, 1.0, 10))
        # df["binned"] = pd.qcut(df['complexity'], 15)

        # plot aspect ratio for each complexity bin
        self.boxplot(df, var="Aspect Ratio", ax=ax1)
        plt.xticks(rotation=90)
        ax1.set_ylabel("Aspect Ratio")
        ax1.set_xlabel("")

        # plot area ratio for each complexity bin
        self.boxplot(df, var="Area Ratio", ax=ax2)
        plt.xticks(rotation=90)
        plt.ylabel("Area Ratio")
        plt.xlabel("Complexity")
        plt.savefig("../plots/complexity_phi_ar.png", bbox_inches="tight")
        # each complexity bin count varies between 138 and 80758
