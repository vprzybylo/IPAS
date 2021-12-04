"""
LOAD CPI DATA
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
class CPI:
    """
    read, flatten, and concatenate cpi date across campaigns
    """

    def __init__(self):
        self.campaigns = []
        self.classifications = []
        self.ars = []  # area ratio
        self.cnt_area = []
        self.a = []  # larger radius
        self.c = []  # smaller radius
        self.Dmax = []  # max diameter
        self.complexity = []
        self.all_campaigns = [
            "AIRS_II",
            "ARM",
            "ATTREX",
            "CRYSTAL_FACE_NASA",
            "CRYSTAL_FACE_UND",
            "ICE_L",
            "IPHEX",
            "ISDAC",
            "MACPEX",
            "MC3E",
            "MIDCIX",
            "MPACE",
            "OLYMPEX",
            "POSIDON",
        ]

    def read_data(self, campaign):
        return pd.read_csv(f"../CPI/databases/no_mask/v1.4.0/{campaign}.csv")

    def concat(self, data):
        """
        flatten a variable across campaigns
        """
        return list(np.concatenate(data).flat)

    def rename_campaigns(self):
        """no underscores for figures"""
        self.campaign = "AIRS II" if self.campaign == "AIRS_II" else self.campaign
        self.campaign = (
            "CRYSTAL FACE NASA"
            if self.campaign == "CRYSTAL_FACE_NASA"
            else self.campaign
        )
        self.campaign = (
            "CRYSTAL FACE UND" if self.campaign == "CRYSTAL_FACE_UND" else self.campaign
        )
        self.campaign = "ICE L" if self.campaign == "ICE_L" else self.campaign

    def lengths(self, df):
        a = []
        c = []
        Dmax = []
        for height, width in zip(df["particle height"] / 2, df["particle width"] / 2):
            a.append(height * 1e-6 if height > width else width * 1e-6)
            c.append(height * 1e-6 if height < width else width * 1e-6)  # [m]
            Dmax.append(height * 2 * 1e-6 if height > width else width * 2 * 1e-6)
        return a, c, Dmax

    def process_campaigns(self):

        for self.campaign in self.all_campaigns:
            df = self.read_data(self.campaign)
            self.cnt_area.append(df["cnt_area"])  # *5.29E-12) # [m2]
            self.ars.append(df["filled_circular_area_ratio"])

            self.rename_campaigns()
            self.campaigns.append(
                [self.campaign] * len(df["filled_circular_area_ratio"])
            )
            self.classifications.append(df["classification"])
            self.complexity.append(df["complexity"])

            a, c, Dmax = self.lengths(df)
            self.a.append(a)
            self.c.append(c)
            self.Dmax.append(Dmax)

    def make_df(self):

        d = {
            "Campaign": self.concat(self.campaigns),
            "Classification": self.concat(self.classifications),
            "Area Ratio": self.concat(self.ars),
            "Contour Area": self.concat(self.cnt_area),
            "a": self.concat(self.a),
            "c": self.concat(self.c),
            "Dmax": self.concat(self.Dmax),
            "Complexity": self.concat(self.complexity),
        }

        self.df_CPI = pd.DataFrame.from_dict(d)

    def remove_baddata(self):
        self.df_CPI = self.df_CPI[self.df_CPI["Area Ratio"] != -999.0]
        self.df_CPI = self.df_CPI[self.df_CPI["a"] != 0.0]
        self.df_CPI = self.df_CPI[self.df_CPI["c"] != 0.0]
        self.df_CPI = self.df_CPI[self.df_CPI["Dmax"] != 0.0]
        self.df_CPI = self.df_CPI[self.df_CPI["Complexity"] != 0.0]
        self.df_CPI = self.df_CPI[self.df_CPI["Complexity"] != -0.0]

        self.df_CPI = self.df_CPI[
            self.df_CPI.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)
        ]
        self.df_CPI.dropna(inplace=True)


@auto_str
class Plot:
    def __init__(self, df_CPI):
        self.df_CPI = df_CPI

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
                df_type = df[var][df["Classification"] == part_type]
                part_type = (
                    "compact irregular" if part_type == "compact_irreg" else part_type
                )
                part_type = (
                    "planar polycrystal"
                    if part_type == "planar_polycrystal"
                    else part_type
                )

                sns.kdeplot(
                    df_type,
                    color=colors_cpi[i],
                    label=part_type,
                    ax=axs[c],
                    linewidth=3,
                )
                no_ylab = [1, 2, 3, 5, 6, 7]
                axs[c].set_ylabel(" " if c in no_ylab else "Density")
                axs[c].set_title(f"{campaign}\n n={len(df):,}")
                axs[c].yaxis.set_major_locator(MaxNLocator(integer=True))
                axs[c].set_xlim(0.0, 1.0)
        y = -0.5
        x = -3.3
        axs[c].legend(
            bbox_to_anchor=(x, y),
            loc="lower center",
            ncol=5,
            title="Particle Classification",
            fontsize=14,
        )
        plt.rcParams["legend.title_fontsize"] = "xx-large"
        plt.savefig(f"../plots/campaign_{var}.png", bbox_inches="tight")
