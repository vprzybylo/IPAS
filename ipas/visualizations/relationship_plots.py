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

        colors = ["#3c1518", "#69140e", "#a44200", "#d58936", "#efd6ac"]
        colors_cpi = ["#0B2B26", "#235347", "#8EB69B", "#DAF1DE", "w"]
        colors_others = ["#072F5F", "#1261A0", "#ADC4DD", "#caf0f8"]

        linewidth = 5
        alpha = 0.7

        D_modes = np.zeros((len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3]))
        m = np.zeros((len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3]))

        for self.phi_idx in self.phi_idxs:
            for self.r_idx in self.r_idxs:
                for nm in range(self.agg_as.shape[3]):
                    self.nm = nm
                    D_modes[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
                        self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    )

                    if mflag == "area":
                        m[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
                            self.mass_spheroid_areas()
                        )
                    else:
                        m[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
                            self.mass_ellipsoid_volumes()
                        )  # kg
                #                     self.ax.scatter(
                #                             D_modes[self.phi_idx, self.r_idx, self.nm] * 1000,
                #                             m[self.phi_idx, self.r_idx, self.nm],
                #                             s=self.nm/3,
                #                             c=colors[self.phi_idx],
                #                         )

                # use a power law to fit a regression line for each IPAS monomer
                # aspect ratio and radius grouping; fitting along increasing nm
                # with 300 aggregates per nm
                x = D_modes[self.phi_idx, self.r_idx, :] * 1000
                y = m[self.phi_idx, self.r_idx, :]
                yfit = self.plot_poly_curve_fits(x, y)
                self.ax.plot(
                    x,
                    yfit,
                    color=colors[self.phi_idx],
                    linewidth=linewidth,
                    label=f"{self.ASPECT_RATIOS[self.phi_idx]} [n=90k]"
                    if self.r_idx == 0
                    else "",
                )

        # CALCULATE AND PLOT MASS FROM CPI IMAGERY
        if mflag == "area":
            particle_types = ["compact_irreg", "agg", "bullet", "column", " "]
            samples = ["99,017", "24,481", "11,433", "16,627", " ", " ", " "]
            cpi_lines = []
            for i, part_type in enumerate(particle_types):
                if part_type == " ":
                    alpha = 0
                    x = 0
                    yfit = 0
                    label = " "
                else:
                    label = f"{part_type} [n={samples[i]}]"
                    df = self.df_CPI[self.df_CPI["classification"] == part_type]
                    part_type = (
                        "compact irregular"
                        if part_type == "compact_irreg"
                        else part_type
                    )
                    df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
                    x = df["Dmax"] * 1000
                    y = self.mass_CPI(df)
                    yfit = self.plot_poly_curve_fits(x, y)
                cpi = self.ax.plot(
                    x, yfit, linewidth=linewidth, color=colors_cpi[i], label=label
                )
                cpi_lines.append(cpi)

        alpha = 1.0
        ### KARRER 2020 aggregates ###
        # dendrites and needles coexist with similar PSD and likeli-hood of aggregation
        # 10E-4 m <= D <= 10E-1 m
        D = np.arange(0.001, 0.1, 0.0001)  # m
        m_aggs = 0.045 * D ** 2.16  # kg
        self.ax.plot(
            D * 1000,
            m_aggs,
            c="#46315C",
            linewidth=linewidth,
            label="K2020 Mix1 [n=105k]",
        )

        ### KARRER 2020 aggregates ###
        #  the monomers with Dmax < 1 mm are columns,
        # while dendrites are taken for larger monomers (”Mix2”)
        # 10E-4 m <= D <= 10E-1 m
        D = np.arange(0.001, 0.1, 0.0001)  # m
        m_aggs = 0.017 * D ** 1.94  # kg
        self.ax.plot(
            D * 1000,
            m_aggs,
            c="#B3A3BA",
            linestyle=":",
            linewidth=linewidth,
            label="K2020 Mix2 [n=105k]",
        )

        ### MITCHELL 1996 ###
        # aggregates of side planes, columns, and bullets
        # 800 mu <= D <= 4500 mu
        D = np.arange(0.0800, 0.4500, 0.0001)  # cm
        m_aggs = (0.0028 * D ** 2.1) * 0.001  # convert from g to kg
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
            label="M90 plate aggregates [n=30]",
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
            label="LH74 mixed aggregates [n=19]",
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

        # LEGEND
        if mflag == "area" and result_rand == True:
            x = 1.1
            y = -2.1
            self.ax.legend(cpi_lines, bbox_to_anchor=(x, y), loc="lower center")
            self.ax.legend(
                bbox_to_anchor=(x, y),
                loc="lower center",
                ncol=3,
                title="   IPAS                                CPI                                      OBSERVATIONS                ",
            )  # fmt: on/off

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

    def bin_D(self, df, linewidth, color, part_type, label):
        """
        bin vt based on D; equal count bins ~9k samples
        find mode in each bin
        plot the mode instead of a scatter plot with all observations
        plot an envelope of min-max value in each bin; shaded
        """

        if part_type == " ":
            # only plot modes
            self.ax.plot(0, 0, alpha=0.0, linewidth=linewidth, color=color, label=label)
        else:
            df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
            df["binned"] = pd.qcut(df["Dmax"], 30)
            groups = df.groupby(df["binned"])
            vt_modes = groups.vt.agg(self.get_modes)
            # only plot modes
            self.ax.plot(
                groups.mean().Dmax * 1000,
                vt_modes,
                alpha=1.0,
                linewidth=linewidth,
                color=color,
                zorder=3,
                label=label,
            )

            # scatter all observations

    #         self.ax.scatter(
    #             df['Dmax'] * 1000,
    #             df['vt'],
    #             alpha=0.05,
    #             color=color,
    #             label=f"{part_type}",
    #         )

    # fill between min and max
    #         self.ax.fill_between(
    #             groups.mean().Dmax * 1000,
    #             y1=groups.vt.min(),
    #             y2=groups.vt.max(),
    #             alpha=0.2,
    #             color=color,
    #         )
    # fill between +/- 2 std
    #         self.ax.fill_between(
    #             groups.mean().Dmax * 1000,
    #             y1=vt_modes - 2*(groups.vt.std()),
    #             y2=vt_modes + 2*(groups.vt.std()),
    #             alpha=0.2,
    #             color=color,
    #         )

    #         self.ax.plot(groups.mean().Dmax * 1000, groups.vt.min(), color=color)
    #         self.ax.plot(groups.mean().Dmax * 1000, groups.vt.max(), color=color)

    def vt_plot(self, title, ylabel, mflag, result_rand):

        colors = ["#3c1518", "#69140e", "#a44200", "#d58936", "#efd6ac"]
        colors_cpi = ["#0B2B26", "#235347", "#8EB69B", "gray", "#DAF1DE"]
        colors_others = ["#072F5F", "#1261A0", "#ADC4DD", "#CAE9F5"]
        linewidth = 4

        # CALCULATE AND PLOT VT FROM CPI IMAGERY
        self.P = 750  # pressure [hPa]
        if mflag == "area":
            # line_style = [":", "-.", "-", "--", "-"]
            particle_types = [
                "agg",
                "bullet",
                "column",
                "planar_polycrystal",
                "compact_irreg",
                " ",
            ]
            T = [-10, -30, -5, -20, -10, 0]
            samples = ["24,481", "11,432", "16,627", "14,363", "99,012", " "]
            for i, part_type in enumerate(particle_types):

                self.T = T[i]
                self.RHO_A = (
                    1.2754 * (self.P / 1000) * (273.15 / (self.T + 273.15))
                )  # air density for a given pressure and temp
                self.dynamic_viscosity()
                df = self.df_CPI[self.df_CPI["classification"] == part_type]
                part_type = (
                    "compact irregular" if part_type == "compact_irreg" else part_type
                )
                df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
                # samples.append(len(df) if part_type != " " else " ")
                Ar = df["area_ratio"]
                # Ap = df['cnt_area']
                m = self.mass_CPI(df)
                D = df["Dmax"]

                if part_type == " ":
                    color = "w"
                    df["vt"] = self.vt
                    label = " "
                    self.bin_D(df, linewidth, color, part_type, label)
                else:
                    X = self.best_number_Heymsfield(Ar, m)
                    Re = self.reynolds_number_Heymsfield(X)
                    self.terminal_velocity(D, Re)

                    #                     X = self.best_number(Ar, m)
                    #                     Re = self.reynolds_number(X)
                    #                     self.terminal_velocity(D, Re)

                    df["vt"] = self.vt
                    color = colors_cpi[i]
                    label = f"{part_type} [n={samples[i]}]"
                    self.bin_D(df, linewidth, color, part_type, label)

        # IPAS
        for self.phi_idx in self.phi_idxs:
            if self.ASPECT_RATIOS[self.phi_idx] < 1.0:
                self.T = -15  # temperature [C] plates
            else:
                self.T = -5  # temperature [C] columns

            self.P = 750  # pressure [hPa]
            self.RHO_A = (
                1.2754 * (self.P / 1000) * (273.15 / (self.T + 273.15))
            )  # air density for a given pressure and temp
            self.dynamic_viscosity()

            for self.r_idx in self.r_idxs:
                for nm in range(self.agg_as.shape[3]):
                    self.nm = nm

                    D_modes = self.get_modes(
                        self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    )

                    if mflag == "area":
                        m = self.mass_spheroid_areas()  #  kg
                    else:
                        m = self.mass_ellipsoid_volumes()  #  kg

                    Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
                    D = self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    # X = self.best_number(Ar, Ap, D, m)
                    X = self.best_number_Heymsfield(Ar, m)

                    X = self.get_modes(X)
                    Re = self.reynolds_number(X)
                    # Re = self.reynolds_number_Heymsfield(X)
                    # self.terminal_velocity_Mitchell_2005(Ar, X)
                    self.terminal_velocity(D_modes, Re)
                    self.ax.scatter(
                        D_modes * 1000,
                        self.vt,
                        s=self.nm / 3,
                        alpha=0.7,
                        zorder=2,
                        c=colors[self.phi_idx],
                        label=f"{self.ASPECT_RATIOS[self.phi_idx]} [n=90k]"
                        if self.r_idx == 0
                        and self.nm == range(self.agg_as.shape[3])[-1]
                        else "",
                    )

        # Zawadski 2010
        D = np.arange(1.0, 8.0, 0.001)  # [mm]

        # Karrer 2020
        D = np.arange(0.001, 0.01, 0.0001)  # m
        # D = np.arange(0.1, 10.0, 0.01)  # mm
        self.ax.plot(
            D * 1000,
            21.739 * D ** 0.580,
            c="#B3A3BA",
            zorder=4,
            linewidth=linewidth,
            label="K2020 Mix1 [n=105k]",
        )

        # Karrer 2020
        D = np.arange(0.001, 0.01, 0.0001)  # m
        # D = np.arange(0.01, 10.0, 0.0001)  # mm
        self.ax.plot(
            D * 1000,
            8.567 * D ** 0.393,
            c="#46315C",
            linewidth=linewidth,
            zorder=4,
            label="K2020 Mix2 [n=105k]",
        )

        # Locatelli and Hobbs 1974
        # aggregates of unrimed radiating assemblages of dendrites
        D = np.arange(2.0, 10.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.8 * D * 0.16,
            c=colors_others[0],
            linewidth=linewidth,
            zorder=3,
            label="LH74 unrimed assemblage dendrite [n=28]",
        )

        # aggregates of unrimed radiating assemblages of
        # plates, sideplanes, bullets, and columns
        D = np.arange(0.2, 3.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.69 * D * 0.41,
            c=colors_others[1],
            linewidth=linewidth,
            zorder=3,
            label="LH74 unrimed assemblage mix [n=31]",
        )

        # aggregates of unrimed sideplanes
        D = np.arange(0.5, 4.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.82 * D * 0.12,
            colors_others[2],
            linewidth=linewidth,
            zorder=3,
            label="LH74 sideplane aggregates [n=23]",
        )

        #         D = np.arange(0.4, 1.2, 0.01)  # mm
        #         self.ax.plot(
        #             D,
        #             0.81 * D * 0.99,
        #             c=colors_others[3],
        #             linewidth=linewidth,
        #             label="LH74 unrimed sideplane [n=10]",
        #         )

        if mflag == "area" and result_rand == True:
            x = 1.1
            y = -2.5

            self.ax.legend(
                bbox_to_anchor=(x, y),
                loc="lower center",
                ncol=3,
                title="    CPI                                                       OBSERVATIONS                                                                IPAS      ",
            )  # fmt: on/off

        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)
        self.ax.set_yscale("log")
        self.ax.set_ylim(0.002, 10.0)
        self.ax.set_xlim([4e-2, 1e2])
        # self.ax.set_xlim(0.0, 10)
        self.ax.set_xscale("log")
        if mflag != "area":
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

        df = self.df_CPI[self.df_CPI["classification"] == "agg"]
        df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
        Ar_CPI = df["area_ratio"]
        Ar_CPI = self.get_modes(Ar_CPI)

        Ap_CPI = df["cnt_area"]
        Ap_CPI = self.get_modes(Ap_CPI)

        df = pd.DataFrame(
            {
                "Ap CPI": Ap_CPI,
                "Ap IPAS": Aps,
                "Ar CPI": Ar_CPI,
                "Ar IPAS": Ars,
                "Vp IPAS": Vps,
                "Ve IPAS": Ves,
                "Vr IPAS": Vrs,
            },
            index=self.ASPECT_RATIOS,
        )
        color = {
            "Ap IPAS": "#98c1d9",
            "Ac IPAS": "#3d5a80",
            "Ar IPAS": "#002347",
            "Vp IPAS": "#F4AC4D",
            "Ve IPAS": "#E26610",
            "Vr IPAS": "#671E14",
            "Ap CPI": "#DAF1DE",
            "Ar CPI": "#235347",
        }
        self.ax = df.plot.bar(rot=0, color=color, ax=self.ax, legend=False, width=0.7)

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
