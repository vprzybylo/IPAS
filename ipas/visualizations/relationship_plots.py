import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mD_vT_relationships as relationships
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import seaborn as sns
import statsmodels
from scipy.optimize import curve_fit


class Plots(relationships.Relationships):
    """
    mass-dimensional and terminal velocity plotting code
    """

    def __init__(
        self,
        ax,
        df_CPI,
        agg_as,
        agg_bs,
        agg_cs,
        phi_idxs,
        r_idxs,
        Aps,
        Acs,
        Vps,
        Ves,
        Dmaxs,
    ):
        super().__init__(
            agg_as, agg_bs, agg_cs, phi_idxs, r_idxs, Aps, Acs, Vps, Ves, Dmaxs
        )
        self.ax = ax
        self.df_CPI = df_CPI

    def plot_poly_curve_fits(self, x, y):
        # fit log(y) = m*log(x) + c
        m, c = np.polyfit(np.log(x), np.log(y), 1)
        yfit = np.exp(m * np.log(x) + c)
        return yfit

    def m_D_plot(self, title, ylabel, mflag="vol", result_rand=False):

        # colors = ["#E0B069", "#B55F56", "#514F51", "#165E6E", "#A0B1BC"]
        colors = ["#3c1518", "#69140e", "#a44200", "#d58936", "#efd6ac"]
        # colors_cpi = ["#c5e1a5", "#B7BF96","#133A1B", "#011B10"]
        colors_cpi = ["#0B2B26", "#235347", "#8EB69B", "#DAF1DE", "w"]
        colors_others = ["#03045e", "#0077b6", "#90e0ef", "#caf0f8"]
        aspect_ratios = [0.01, 0.10, 1.00, 10.0, 50.0]
        linewidth = 5
        alpha = 0.7
        D_modes = np.zeros((len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3]))
        m_area_modes = np.zeros(
            (len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3])
        )
        m_vol_modes = np.zeros(
            (len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3])
        )

        for self.phi_idx in self.phi_idxs:
            for self.r_idx in self.r_idxs:
                for nm in range(self.agg_as.shape[3]):
                    self.nm = nm
                    D_modes[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
                        self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    )
                    m_spheroid_area = self.mass_spheroid_areas()
                    m_area_modes[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
                        m_spheroid_area
                    )

                    m_vol_modes[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
                        self.mass_ellipsoid_volumes()
                    )  # kg

                    m = m_area_modes if mflag == "area" else m_vol_modes
                #                     sc_IPAS = self.ax.scatter(
                #                             D_modes[self.phi_idx, self.r_idx, self.nm] * 1000,
                #                             m[self.phi_idx, self.r_idx, self.nm],
                #                             s=self.nm/3,
                #                             c=colors[self.phi_idx],
                #                         )

                x = D_modes[self.phi_idx, self.r_idx, :] * 1000
                m = m_area_modes if mflag == "area" else m_vol_modes
                y = m[self.phi_idx, self.r_idx, :]
                yfit = self.plot_poly_curve_fits(x, y)
                self.ax.plot(
                    x,
                    yfit,
                    color=colors[self.phi_idx],
                    linewidth=linewidth,
                    label=aspect_ratios[self.phi_idx] if self.r_idx == 0 else "",
                )

        # CPI
        if mflag == "area":
            line_style = [":", "-.", "-", "--", " "]
            particle_types = ["compact_irreg", "agg", "bullet", "column", " "]
            cpi_lines = []
            for i, part_type in enumerate(particle_types):
                if part_type == " ":
                    alpha = 0
                    x = 0
                    yfit = 0
                else:
                    df = self.df_CPI[self.df_CPI["classification"] == part_type]
                    df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
                    x = df["a"] * 1000
                    y = self.mass_CPI(df)
                    yfit = self.plot_poly_curve_fits(x, y)
                cpi = self.ax.plot(
                    x,
                    yfit,
                    linewidth=linewidth,
                    linestyle=line_style[i],
                    color=colors_cpi[i],
                    label=f"{part_type}",
                )
                cpi_lines.append(cpi)

        alpha = 1.0
        ### KARRER 2020 aggregates ###
        # dendrites and needles coexist with similar PSD and likeli-hood of aggregation
        # 10E-4 m <= D <= 10E-1 m
        D = np.arange(0.0001, 0.1, 0.0001)  # mu
        m_aggs = 0.045 * D ** 2.16  # kg
        self.ax.plot(
            D * 1000,
            m_aggs,
            c="purple",
            linestyle=":",
            linewidth=linewidth,
            alpha=alpha,
            label="K2020 Mix1",
        )

        ### KARRER 2020 aggregates ###
        #  the monomers with Dmax < 1 mm are columns,
        # while dendrites are taken for larger monomers (”Mix2”)
        # 10E-4 m <= D <= 10E-1 m
        D = np.arange(0.0001, 0.1, 0.0001)  # mu
        m_aggs = 0.017 * D ** 1.94  # kg
        self.ax.plot(
            D * 1000,
            m_aggs,
            c="indigo",
            linewidth=linewidth,
            alpha=alpha,
            label="K2020 Mix2",
        )

        ### MITCHELL 1996 ###
        # aggregates of side planes, columns, and bullets
        # 800 mu <= D <= 4500 mu
        D = np.arange(0.0800, 0.4500, 0.0001)  # mu
        m_aggs = (0.0028 * D ** 2.1) * 0.001  # kg
        self.ax.plot(
            D * 10,
            m_aggs,
            c=colors_others[0],
            linewidth=linewidth,
            alpha=alpha,
            label="M96 aggregates",
        )

        #  aggregates of radiating assemblages of plates
        D = np.arange(0.8, 7.7, 0.0001)  # mm
        m_aggs = (0.023 * D ** 1.8) * 1e-6
        self.ax.plot(
            D,
            m_aggs,
            c=colors_others[1],
            linewidth=linewidth,
            linestyle="--",
            label="M90 plate aggregates",
            alpha=alpha,
        )

        ### Locatellii and Hobbs 1974 ###
        D = np.arange(1.0, 3.0, 0.0001)  # mm
        m = (0.037 * D ** 1.9) * 1e-6  # mg
        self.ax.plot(
            D,
            m,
            c=colors_others[2],
            linewidth=linewidth,
            linestyle="--",
            label="LH74 mixed aggregates",
            alpha=alpha,
        )

        #         D = np.arange(2, 10, 0.0001)  # mm
        #         m = (0.073 * D ** 1.4) * 1e-6
        #         self.ax.plot(
        #             D,
        #             m,
        #             c=colors_others[3],
        #             linewidth=linewidth,
        #             linestyle="--",
        #             label="LH74 dendritic aggregates",
        #             alpha=alpha,
        #         )

        if mflag == "area" and result_rand == True:
            x = 1.1
            y = -2.1
            self.ax.legend(cpi_lines, bbox_to_anchor=(x, y), loc="lower center")
            self.ax.legend(
                bbox_to_anchor=(x, y),
                loc="lower center",
                ncol=3,
                title="         IPAS                           CPI                          OBSERVATIONS                      ",
            )

        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)
        self.ax.set_yscale("log")
        self.ax.set_xscale("log")
        if mflag != "area":
            self.ax.set_xlabel("$D_{max}$ [mm]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_ylim([1e-12, 2e-1])
        self.ax.set_xlim([6e-3, 3e2])
        self.ax.set_title(title)

    def vt_plot(self, title, ylabel):

        # colors = ["#E0B069", "#B55F56", "#514F51", "#165E6E", "#A0B1BC"]

        for self.phi_idx in self.phi_idxs:
            for self.r_idx in self.r_idxs:
                for nm in range(99):
                    self.nm = nm
                    # print(self.phi_idx, self.nm)

                    self.dynamic_viscosity()
                    D = self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    D = self.get_modes(D)

                    Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
                    Ap = self.Aps[self.phi_idx, self.r_idx, :, self.nm]
                    Ac = self.Acs[self.phi_idx, self.r_idx, :, self.nm]
                    # m_spheroid_area = self.mass_spheroid_areas(Ar)

                    Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
                    # Vp = self.Vps[self.phi_idx, self.r_idx, :, self.nm]
                    m_ellipsoid_vol = self.mass_ellipsoid_volumes(Vr)  #  kg
                    X = self.best_number(Ar, D, Ap, Ac, Ar, m_ellipsoid_vol)
                    X = self.get_modes(X)
                    Re = self.reynolds_number(X)
                    # X = self.best_number_Mitchell(Ar)

                    # self.terminal_velocity_Mitchell_2005(Ar, X)
                    self.terminal_velocity_Mitchell(D, Re)
                    # self.terminal_velocity_Mitchell(D, Re_area)

                    #                 plt.plot(
                    #                     D,
                    #                     700*D,
                    #                     c='g',
                    #                     linewidth=3,
                    #                     label='Lin ice'
                    #                 )

                    #                 plt.plot(
                    #                     D,
                    #                     11.72 * D ** 0.41,
                    #                     c='lightgreen',
                    #                     linewidth=3,
                    #                     label='Lin snow'
                    #                 )

                    D = D * 1000  # mm
        self.ax.legend(title="Monomer\nAspect Ratio", loc="lower right")
        # Locatelli and Hobbs 1974
        # aggregates of unrimed radiating assemblages of dendrites
        D = np.arange(2.0, 10.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.8 * D * 0.16,
            c="olivedrab",
            linewidth=3,
            label="LH unrimed assemblage dendrite",
        )

        # aggregates of unrimed radiating assemblages of
        # plates, sideplanes, bullets, and columns
        D = np.arange(0.2, 3.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.069 * D * 0.41,
            c="darkolivegreen",
            linewidth=3,
            label="LH unrimed assemblage mix",
        )

        # aggregates of unrimed sideplanes
        D = np.arange(0.5, 4.0, 0.01)  # mm
        self.ax.plot(
            D, 0.082 * D * 0.12, c="khaki", linewidth=3, label="LH unrimed sideplanes"
        )

        # Zawadski 2010
        D = np.arange(0.10, 8.0, 0.01)  # mm
        self.ax.plot(D, 0.069 * D * 0.21, c="k", linewidth=3, label="Zawadski 2010")

        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)
        self.ax.set_yscale("log")
        # self.ax.set_ylim(0.01, 15.0)
        # self.ax.set_xlim(3E-2, 3E2)
        # self.ax.set_xlim(0.0, 10)
        self.ax.set_xscale("log")
        self.ax.set_xlabel("$D_{max}$ [mm]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)

    def area_plot(self, title, xlabel, nm):

        Vps, Ves, Vrs, Aps, Acs, Ars = [], [], [], [], [], []
        for self.phi_idx in self.phi_idxs:

            Ap = self.get_modes(self.Aps[self.phi_idx, self.r_idx, :, self.nm])
            Aps.append(Ap)

            Ac = self.get_modes(self.Acs[self.phi_idx, self.r_idx, :, self.nm])
            Acs.append(Ac)

            Ar = self.get_modes(self.Ars[self.phi_idx, self.r_idx, :, self.nm])
            Ars.append(Ar)

            Vp = self.Vps[self.phi_idx, self.r_idx, 0, self.nm]
            Vps.append(Vp)

            Ve = self.get_modes(self.Ves[self.phi_idx, self.r_idx, :, self.nm])
            Ves.append(Ve)

            Vr = self.get_modes(self.Vrs[self.phi_idx, self.r_idx, :, self.nm])
            Vrs.append(Vr)

        df = pd.DataFrame(
            {"Ap": Aps, "Ac": Acs, "Ar": Ars, "Vp": Vps, "Ve": Ves, "Vr": Vrs},
            index=self.ASPECT_RATIOS,
        )
        color = {
            "Ap": "#98c1d9",
            "Ac": "#3d5a80",
            "Ar": "#002347",
            "Vp": "#F4AC4D",
            "Ve": "#E26610",
            "Vr": "#671E14",
        }
        self.ax = df.plot.bar(rot=0, color=color, ax=self.ax, legend=False)

        # self.ax.set_ylim([1E-4, 1E11])
        self.ax.set_yscale("log")
        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Area [$\mu m^2$] or Volume [$\mu m^3$]")
        self.ax.set_title(title)

    def mass_plot(self, title, xlabel):

        m_areas, m_vols = [], []
        for self.phi_idx in self.phi_idxs:
            Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
            m_ellipsoid_vol = self.mass_ellipsoid_volumes(Vr)  #  kg
            m_ellipsoid_vol = self.get_modes(m_ellipsoid_vol)
            m_vols.append(m_ellipsoid_vol)

            Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
            m_spheroid_area = self.mass_spheroid_areas(Ar)  #  kg
            m_spheroid_area = self.get_modes(m_spheroid_area)
            m_areas.append(m_spheroid_area)

        df = pd.DataFrame({"Area": m_areas, "Volume": m_vols}, index=self.ASPECT_RATIOS)
        color = {"Area": "#3d5a80", "Volume": "#E26610"}
        self.ax = df.plot.bar(rot=0, color=color, width=0.7, ax=self.ax, legend=False)

        self.ax.set_yscale("log")
        self.ax.grid(which="major")
        # self.ax.grid(which="minor")
        # self.ax.grid(True)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Mass [kg]", color="maroon")
        self.ax.set_title(title)

    def best_number_plot(self, title, xlabel):

        X_areas, X_vols = [], []
        for self.phi_idx in self.phi_idxs:

            self.dynamic_viscosity()
            D = self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]

            # best number using area
            Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
            Ap = self.Aps[self.phi_idx, self.r_idx, :, self.nm]
            m_spheroid_area = self.mass_spheroid_areas(Ar)
            X_area = self.best_number(Ar, D, Ap, m_spheroid_area)  # shape of 300
            X_area = self.get_modes(X_area)
            X_areas.append(X_area)

            # best number using volume
            Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
            Vp = self.Vps[self.phi_idx, self.r_idx, :, self.nm]
            m_ellipsoid_vol = self.mass_ellipsoid_volumes(Vr)  #  kg
            X_vol = self.best_number(Vr, D, Vp, m_ellipsoid_vol)
            X_vol = self.get_modes(X_vol)
            X_vols.append(X_vol)

        df = pd.DataFrame({"Area": X_areas, "Volume": X_vols}, index=self.ASPECT_RATIOS)
        color = {"Area": "#3d5a80", "Volume": "#E26610"}
        self.ax = df.plot.bar(rot=0, color=color, width=0.7, ax=self.ax, legend=False)

        self.ax.set_yscale("log")
        self.ax.grid(which="major")
        # self.ax.grid(which="minor")
        # self.ax.grid(True)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Best Number", color="maroon")
        self.ax.set_title(title)

    def density_plot(self, title, xlabel):

        rhoi_area, rhoi_vol = [], []
        for self.phi_idx in self.phi_idxs:

            Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
            rho_i = self.get_modes(self.RHO_B * Vr)
            rhoi_vol.append(rho_i)

            Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
            rho_i = self.get_modes(self.RHO_B * Ar)
            rhoi_area.append(rho_i)

        df = pd.DataFrame(
            {"Area": rhoi_area, "Volume": rhoi_vol}, index=self.ASPECT_RATIOS
        )
        color = {"Area": "#3d5a80", "Volume": "#E26610"}
        self.ax = df.plot.bar(rot=0, color=color, width=0.7, ax=self.ax, legend=False)

        # self.ax.set_yscale('log')
        self.ax.grid(which="major")
        # self.ax.grid(which="minor")
        # self.ax.grid(True)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Density [$kg/m^3$]", color="maroon")
        self.ax.set_title(title)
