"""
Module for holding all plotting code for MON-MON collection
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter


def aspect_ratios_Na(
    neg_error_flat,
    pos_error_flat,
    chs_flat,
    neg_error_rand,
    pos_error_rand,
    chs_rand,
    save_fig=False,
):
    """plot IPAS characteristic aspect ratio for
    different number of aggregates to see consistency in trends
    """
    s = 25  # scatter marker size
    Na = [100, 300, 500, 1000]

    colors_p = ["#D5FFAD", "#79E297", "#328581", "#111A7E"]
    colors_c = ["#F5B841", "#DA9D58", "#B05E2F", "#3B0210"]

    cmap = cm.get_cmap("Spectral", 8)
    colors = []
    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        colors.append(mpl.colors.rgb2hex(rgba))
    colors_p = colors[:4]
    colors_p.reverse()
    colors_c = colors[4:]
    colors_c

    phio = np.logspace(-2, 2, num=20, dtype=None)
    phio_p = phio[:10]
    phio_c = phio[10:]

    alpha = 0.03

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.25, hspace=0.1)
    ###############################################
    cols = []
    plates = []
    for n in range(len(Na)):
        ax1.fill_between(
            phio_p,
            neg_error_flat[:10, n],
            pos_error_flat[:10, n],
            color=colors_p[n],
            alpha=alpha,
        )
        plate = ax1.scatter(
            phio_p, chs_flat[:10, n], c=colors_p[n], label="$n_a$={}".format(Na[n]), s=s
        )
        ax1.fill_between(
            phio_c,
            neg_error_flat[10:, n],
            pos_error_flat[10:, n],
            color=colors_c[n],
            alpha=alpha,
        )
        col = ax1.scatter(
            phio_c, chs_flat[10:, n], c=colors_c[n], label="$n_a$={}".format(Na[n]), s=s
        )
        cols.append(col)
        plates.append(plate)
        if n == 3:
            # plot regression lines QH
            m, b = np.polyfit(np.log(phio_c), np.log(chs_flat[10:, n]), 1)
            y_fit = np.exp(m * np.log(phio_c) + b)
            ax1.plot(phio_c, y_fit, color="navy")
            m, b = np.polyfit(np.log(phio_p), np.log(chs_flat[:10, n]), 1)
            y_fit = np.exp(m * np.log(phio_p) + b)  # calculate the fitted values of y
            ax1.plot(phio_p, y_fit, color="darkred")

    ax1.plot(np.logspace(-2, 0), np.logspace(-2, 0), "k", linestyle="dashed", alpha=0.5)
    ax1.plot(np.logspace(0, 2), np.logspace(0, -2), "k", linestyle="dashed", alpha=0.5)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim([0.01, 1.0])
    ax1.set_xlim([0.01, 100.0])
    ax1.grid()
    ax1.set_ylabel("Aggregate Aspect Ratio")
    ax1.set_title("Quasi-Horizontal Orientation")
    ax1.set_xlabel("Monomer Aspect Ratio")
    ###############################################

    for n in range(len(Na)):
        ax2.fill_between(
            phio_p,
            neg_error_rand[:10, n],
            pos_error_rand[:10, n],
            alpha=0.05,
            color=colors_p[n],
        )
        ax2.scatter(phio_p, chs_rand[:10, n], c=colors_p[n], s=s)
        ax2.fill_between(
            phio_c,
            neg_error_rand[10:, n],
            pos_error_rand[10:, n],
            alpha=alpha,
            color=colors_c[n],
        )
        ax2.scatter(phio_c, chs_rand[10:, n], c=colors_c[n], s=s)
        if n == 3:
            # plot regression lines - random
            m, b = np.polyfit(np.log(phio_p), np.log(chs_rand[:10, n]), 1)
            y_fit = np.exp(m * np.log(phio_p) + b)  # calculate the fitted values of y
            plt.plot(phio_p, y_fit, color="darkred")

            m, b = np.polyfit(np.log(phio_c), np.log(chs_rand[10:, n]), 1)
            # print(m,b)
            y_fit = np.exp(m * np.log(phio_c) + b)  # calculate the fitted values of y
            plt.plot(phio_c, y_fit, color="navy")

    # legend
    leg_labels = ["$n_a$={}".format(Na[n]) for n in range(len(Na))]
    leg1 = ax2.legend(
        plates,
        leg_labels,
        title="Plates",
        bbox_to_anchor=(1.5, 0.5),
        loc="lower right",
        fontsize=14,
        title_fontsize=14,
    )
    leg2 = ax2.legend(
        cols,
        leg_labels,
        title="Columns",
        bbox_to_anchor=(1.5, 0),
        loc="lower right",
        fontsize=14,
        title_fontsize=14,
    )
    ax2.add_artist(leg1)
    ax2.add_artist(leg2)
    ax2.plot(np.logspace(-2, 2), np.logspace(-2, 2), "k", linestyle="dashed", alpha=0.5)
    ax2.plot(np.logspace(0, 2), np.logspace(0, -2), "k", linestyle="dashed", alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim([0.01, 1.0])
    ax2.set_xlim([0.01, 100.0])
    ax2.grid()
    ax2.set_xlabel("Monomer Aspect Ratio")
    ax2.set_title("Random Orientation")

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if save_fig:
        plt.savefig("../plots/partI_aspectratios_Na.pdf", bbox_inches="tight")


def axislengths_aspectratios(
    phio_p,
    phio_c,
    mono_as_p,
    mono_cs_p,
    mono_as_c,
    mono_cs_c,
    avg_phi_flat,
    neg_error_flat_phis,
    pos_error_flat_phis,
    avg_as_flat,
    neg_error_flat_as,
    pos_error_flat_as,
    avg_cs_flat,
    neg_error_flat_cs,
    pos_error_flat_cs,
    avg_phi_rand,
    neg_error_rand_phis,
    pos_error_rand_phis,
    avg_as_rand,
    neg_error_rand_as,
    pos_error_rand_as,
    avg_cs_rand,
    neg_error_rand_cs,
    pos_error_rand_cs,
    delta_flat_plates_major,
    delta_flat_plates_minor,
    delta_flat_columns_major,
    delta_flat_columns_minor,
    delta_rand_plates_major,
    delta_rand_plates_minor,
    delta_rand_columns_major,
    delta_rand_columns_minor,
    save_fig=True,
    alpha=0.3,
):
    """
    plot axis lengths and aspect ratio for each orientation with respect to monomer aspect ratio
    averaged over 4 simulations with Na=300
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey=False, figsize=(16, 10)
    )
    plt.subplots_adjust(wspace=0.25, hspace=0.1)

    # Flat
    ax1.plot(phio_p, avg_as_flat[:10], "b")
    ax1.fill_between(
        phio_p, neg_error_flat_as[:10], pos_error_flat_as[:10], color="b", alpha=alpha
    )
    ax1.plot(phio_p, mono_as_p, "b", linestyle="dotted")
    ax1.plot(phio_p, avg_cs_flat[:10], "orange")
    ax1.fill_between(
        phio_p,
        neg_error_flat_cs[:10],
        pos_error_flat_cs[:10],
        color="orange",
        alpha=alpha,
    )
    ax1.plot(phio_p, mono_cs_p, "darkorange", linestyle="dotted")
    ax1.plot(phio_p, avg_phi_flat[:10], "g")
    ax1.fill_between(
        phio_p,
        neg_error_flat_phis[:10],
        pos_error_flat_phis[:10],
        color="g",
        alpha=alpha,
    )
    ax1.plot(phio_c, avg_as_flat[10:], "b")
    ax1.fill_between(
        phio_c, neg_error_flat_as[10:], pos_error_flat_as[10:], color="b", alpha=alpha
    )
    ax1.plot(phio_c, mono_cs_c, "b", linestyle="dotted")
    ax1.plot(phio_c, avg_cs_flat[10:], "orange")
    ax1.fill_between(
        phio_c,
        neg_error_flat_cs[10:],
        pos_error_flat_cs[10:],
        color="orange",
        alpha=alpha,
    )
    ax1.plot(phio_c, mono_as_c, "darkorange", linestyle="dotted")
    ax1.plot(phio_c, avg_phi_flat[10:], "g")
    ax1.fill_between(
        phio_c,
        neg_error_flat_phis[10:],
        pos_error_flat_phis[10:],
        color="g",
        alpha=alpha,
    )

    ax1.set_xlim([0.01, 100.0])
    ax1.set_ylim([0.005, 350.00])
    ax1.grid()
    ax1.set_ylabel("Axis Lengths and Aspect Ratio")
    ax1.set_title("Quasi-Horizontal Orientation")
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    ###############################################################
    # Random
    ax2.plot(phio_p, avg_as_rand[:10], "b", label="aggregate major axis")
    ax2.fill_between(
        phio_p, neg_error_rand_as[:10], pos_error_rand_as[:10], color="b", alpha=alpha
    )
    ax2.plot(phio_p, mono_as_p, "b", linestyle="dotted", label="monomer major axis")
    ax2.plot(phio_p, avg_cs_rand[:10], "orange", label="aggregate minor axis")
    ax2.fill_between(
        phio_p,
        neg_error_rand_cs[:10],
        pos_error_rand_cs[:10],
        color="orange",
        alpha=alpha,
    )
    ax2.plot(
        phio_p, mono_cs_p, "darkorange", linestyle="dotted", label="monomer minor axis"
    )
    ax2.plot(phio_p, avg_phi_rand[:10], "g", label="aggregate aspect ratio")
    ax2.fill_between(
        phio_p,
        neg_error_rand_phis[:10],
        pos_error_rand_phis[:10],
        color="g",
        alpha=alpha,
    )
    ax2.plot(phio_c, avg_as_rand[10:], "b")
    ax2.fill_between(
        phio_c, neg_error_rand_as[10:], pos_error_rand_as[10:], color="b", alpha=alpha
    )
    ax2.plot(phio_c, mono_cs_c, "b", linestyle="dotted")
    ax2.plot(phio_c, avg_cs_rand[10:], "orange")
    ax2.fill_between(
        phio_c,
        neg_error_rand_cs[10:],
        pos_error_rand_cs[10:],
        color="orange",
        alpha=alpha,
    )
    ax2.plot(phio_c, mono_as_c, "darkorange", linestyle="dotted")
    ax2.plot(phio_c, avg_phi_rand[10:], "g")
    ax2.fill_between(
        phio_c,
        neg_error_rand_phis[10:],
        pos_error_rand_phis[10:],
        color="g",
        alpha=alpha,
    )
    ax2.grid()
    ax2.set_xlim([0.01, 100.0])
    ax2.set_ylim([0.005, 350.00])
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax2.set_title("Random Orientation")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ###############################################################
    # CHANGE IN AXIS LENGTHS
    # plates flat
    ax3.plot(phio_p, delta_flat_plates_major * 100, "b", label="Major axis")
    ax3.plot(phio_p, delta_flat_plates_minor * 100, "orange", label="Minor axis")
    ax3.plot(phio_c, delta_flat_columns_major * 100, "b")
    ax3.plot(phio_c, delta_flat_columns_minor * 100, "orange")
    ax3.legend()
    ax3.grid()
    ax3.set_ylabel("Change in Axis Length (%)")
    ax3.set_xlabel("Monomer Aspect Ratio")
    ax3.set_xlim([0.01, 100.0])
    # ax3.set_ylim([0,250])
    ax3.set_ylim([10.0, 7000])
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    ###############################################################
    # plates random
    ax4.plot(phio_p, delta_rand_plates_major * 100, "b", label="Major axis")
    ax4.plot(phio_p, delta_rand_plates_minor * 100, "orange", label="Minor axis")
    ax4.plot(phio_c, delta_rand_columns_major * 100, "b")
    ax4.plot(phio_c, delta_rand_columns_minor * 100, "orange")
    ax4.legend()
    ax4.grid()
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlim([0.01, 100.0])
    ax4.set_ylim([10.0, 7000])
    ax4.set_xlabel("Monomer Aspect Ratio")

    for ax in [ax1, ax2, ax3, ax4]:
        if ax == ax1 or ax == ax2:
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if ax == ax3 or ax == ax4:
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_major_formatter().set_useOffset(False)

    if save_fig:
        plt.savefig("../plots/partI_axislengths_aspectratios.pdf", bbox_inches="tight")
