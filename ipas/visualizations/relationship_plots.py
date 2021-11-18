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
        self.obs_names = {
            "KMix1": "#46315C",
            "KMix2": "#B3A3BA",
            "M96": "#072F5F",
            "M90": "#1261A0",
            "LH74 mix": "#ADC4DD",
            "LH74 dendrite": "#CAE9F5",
            "LH74 sideplane agg": "darkblue",
            "Z": "k",
        }
        self.colors = ["#3c1518", "#69140e", "#a44200", "#d58936", "#efd6ac"]  # ipas
        self.colors_cpi = ["#132010", "#396031", "#4c8042", "#7fb375", "#9fc697"]

        # colors_others = ["#072F5F", "#1261A0", "#ADC4DD", "#caf0f8"]

        self.linewidth = 4
        self.particle_types = [
            "agg",
            "bullet",
            "column",
            "planar_polycrystal",
            "compact_irreg",
        ]
        self.samples = ["24,481", "11,432", "16,627", "14,363", "99,012", " "]
        self.P = 750  # pressure [hPa]

    def plot_poly_curve_fits(self, x, y):
        """
        fit regression curve in log-log space to IPAS data
        """
        # fit log(y) = m*log(x) + c

        # add catch for x<0 for planar
        x1 = x[x > 0]
        y = y[x > 0]
        m, c = np.polyfit(np.log(x1), np.log(y), 1)
        yfit = np.exp(m * np.log(x1) + c)
        return x1, yfit

    def m_ipas(self, mflag):
        """
        calculate and plot mass of ipas particles
        """

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

                print(m)
                # use a power law to fit a regression line for each IPAS monomer
                # aspect ratio and radius grouping; fitting along increasing nm
                # with 300 aggregates per nm
                x = D_modes[self.phi_idx, self.r_idx, :] * 1000
                y = m[self.phi_idx, self.r_idx, :]
                x1, yfit = self.plot_poly_curve_fits(x, y)
                self.ax.plot(
                    x1,
                    yfit,
                    color=self.colors[self.phi_idx],
                    linewidth=self.linewidth,
                    label=f"{self.ASPECT_RATIOS[self.phi_idx]} [n=90k]"
                    if self.r_idx == 0
                    else "",
                )

    def m_cpi(self):
        """calculate and plot mass from cpi observed particles"""

        cpi_lines = []
        for i, part_type in enumerate(self.particle_types):
            if part_type == " ":
                x = 0
                yfit = 0
                label = " "
            else:
                label = f"{part_type} [n={self.samples[i]}]"
                df = self.df_CPI[self.df_CPI["classification"] == part_type]
                part_type = (
                    "compact irregular" if part_type == "compact_irreg" else part_type
                )
                df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
                x = df["Dmax"] * 1000
                y = self.mass_CPI(df)
                x1, yfit = self.plot_poly_curve_fits(x, y)
            cpi = self.ax.plot(
                x1,
                yfit,
                linewidth=self.linewidth,
                color=self.colors_cpi[i],
                label=label,
            )
            cpi_lines.append(cpi)

        return cpi_lines

    def m_D_plot(self, title, ylabel, mflag="vol", result_rand=False):
        """
        calculate and plot the mass of ice particles from IPAS and CPI data
        other empirical estimates included below
        """

        self.m_ipas(mflag)
        if mflag == "area":
            cpi_lines = self.m_cpi()

        ### KARRER 2020 aggregates ###
        # dendrites and needles coexist with similar PSD and likeli-hood of aggregation
        # 10E-4 m <= D <= 10E-1 m
        D = np.arange(0.0001, 0.01, 0.00001)  # m
        m_aggs = 0.045 * D ** 2.16  # kg
        self.ax.plot(
            D * 1000,
            m_aggs,
            c=self.obs_names["KMix1"],
            linewidth=self.linewidth,
            label="K2020 Mix1 [n=105k]",
        )

        ### KARRER 2020 aggregates ###
        #  the monomers with Dmax < 1 mm are columns,
        # while dendrites are taken for larger monomers (”Mix2”)
        D = np.arange(0.0001, 0.01, 0.00001)  # m
        m_aggs = 0.017 * D ** 1.94  # kg
        self.ax.plot(
            D * 1000,
            m_aggs,
            c=self.obs_names["KMix2"],
            linestyle=":",
            linewidth=self.linewidth,
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
            c=self.obs_names["M96"],
            linewidth=self.linewidth,
            label="M96 aggregates",
        )

        #  aggregates of radiating assemblages of plates
        D = np.arange(0.8, 7.7, 0.0001)  # mm
        m_aggs = (0.023 * D ** 1.8) * 1e-6
        self.ax.plot(
            D,
            m_aggs,
            c=self.obs_names["M90"],
            linewidth=self.linewidth,
            label="M90 plate aggregates [n=30]",
        )

        ### Locatellii and Hobbs 1974 ###
        D = np.arange(1.0, 3.0, 0.0001)  # mm
        m = (0.037 * D ** 1.9) * 1e-6  # mg
        self.ax.plot(
            D,
            m,
            c=self.obs_names["LH74 mix"],
            linewidth=self.linewidth,
            label="LH74 mixed aggregates [n=19]",
        )

        #         D = np.arange(2, 10, 0.0001)  # mm
        #         m = (0.073 * D ** 1.4) * 1e-6
        #         self.ax.plot(
        #             D,
        #             m,
        #             c=self.colors_others[3],
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

    def bin_D(self, df, color, part_type, label):
        """
        bin vt based on D; equal count bins ~9k samples
        find mode in each bin
        plot the mode instead of a scatter plot with all observations
        plot an envelope of min-max value in each bin; shaded
        """
        if part_type == " ":
            # only plot modes
            self.ax.plot(0, 0, alpha=0.0, color=color, label=label)
        else:
            df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
            df["binned"] = pd.qcut(df["Dmax"], 30)
            groups = df.groupby(df["binned"])
            vt_modes = groups.vt.agg(self.get_modes)
            # only plot modes
            self.ax.plot(
                groups.mean().Dmax * 1000,
                vt_modes,
                linewidth=self.linewidth,
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

    def cpi_vt(self, study, result_rand, ylabel):
        """
        calculate and plot terminal velocities of CPI observed ice particles
        """
        for i, part_type in enumerate(self.particle_types):
            T = [-10, -30, -5, -20, -10, 0]
            self.RHO_A = (
                1.2754 * (self.P / 1000) * (273.15 / (T[i] + 273.15))
            )  # air density for a given pressure and temp
            self.dynamic_viscosity(T[i])
            df = self.df_CPI[self.df_CPI["classification"] == part_type]
            part_type = (
                "compact irregular" if part_type == "compact_irreg" else part_type
            )
            df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
            # samples.append(len(df) if part_type != " " else " ")
            if part_type == " ":
                color = "w"
                df["vt"] = self.vt
                label = " "
                self.bin_D(df, color, part_type, label)
            else:
                Ar = df["area_ratio"]
                Ap = df["cnt_area"]
                # print('CPI', Ap)
                m = self.mass_CPI(df)
                D = df["Dmax"]
                if study == "Mitchell":
                    Xs = self.best_number_Mitchell(Ar, Ap, D, m)
                    Res = self.reynolds_number_Mitchell(Xs)
                    self.terminal_velocity(D, Res)
                if study == "Heymsfield":
                    Xs = self.best_number_Heymsfield(Ar, m)
                    Res = self.reynolds_number_Heymsfield(Xs)
                    self.terminal_velocity(D, Res)
                if study == "Heymsfield2002":
                    self.vt_Heymsfield(Ar, D)
                df["vt"] = self.vt

                color = self.colors_cpi[i]
                label = f"{part_type} [n={self.samples[i]}]"
                self.bin_D(df, color, part_type, label)

    def ipas_vt(self, mflag, study, result_rand, ylabel):
        """
        calculate and plot ipas terminal fall velocities
        as a function of aspect ratio and orientation
        """

        for self.phi_idx in self.phi_idxs:
            if self.ASPECT_RATIOS[self.phi_idx] < 1.0:
                T = -15
            else:
                T = -5
            self.RHO_A = (
                1.2754 * (self.P / 1000) * (273.15 / (T + 273.15))
            )  # air density for a given pressure and temp
            self.dynamic_viscosity(T)

            for self.r_idx in self.r_idxs:
                for nm in range(self.agg_as.shape[3]):
                    self.nm = nm

                    if mflag == "area":
                        m = self.mass_spheroid_areas()  #  kg
                    else:
                        m = self.mass_ellipsoid_volumes()  #  kg

                    Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
                    D = self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    if study == "Mitchell":
                        Ap = self.Aps[self.phi_idx, self.r_idx, :, self.nm]
                        # print('IPAS', Ap)
                        X = self.best_number_Mitchell(Ar, Ap, D, m)
                        # X = self.get_modes(X)
                        Re = self.reynolds_number_Mitchell(X)
                        self.terminal_velocity(D, Re)
                    if study == "Heymsfield":
                        X = self.best_number_Heymsfield(Ar, m)
                        # X = self.get_modes(X)
                        Re = self.reynolds_number_Heymsfield(X)
                        self.terminal_velocity(D, Re)

                    if study == "Heymsfield2002":
                        self.vt_Heymsfield(Ar, D)

                    vt = self.get_modes(self.vt)
                    D_modes = self.get_modes(
                        self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    )

                    self.ax.scatter(
                        D_modes * 1000,
                        vt,
                        s=self.nm / 3,
                        alpha=0.7,
                        zorder=2,
                        c=self.colors[self.phi_idx],
                        label=f"{self.ASPECT_RATIOS[self.phi_idx]} [n=90k]"
                        if self.r_idx == 0
                        and self.nm == range(self.agg_as.shape[3])[-1]
                        else "",
                    )

    def vt_plot(self, title, ylabel, mflag, study, result_rand):
        """
        plot all vt relationships
        calls ipas and cpi vt methods
        additionally holds empirical relationships from other studies
        """

        self.ipas_vt(mflag, study, result_rand, ylabel)
        if mflag == "area":
            self.cpi_vt(study, result_rand, ylabel)

        # Zawadski 2010
        D = np.arange(1.0, 8.0, 0.001)  # [mm]
        self.ax.plot(
            D,
            0.69 * D ** 0.21,
            c=self.obs_names["Z"],
            zorder=4,
            linewidth=self.linewidth,
            label="Z10 [n=16,324]",
        )

        # Karrer 2020
        D = np.arange(0.0001, 0.01, 0.0001)  # m
        # D = np.arange(0.1, 10.0, 0.01)  # mm
        self.ax.plot(
            D * 1000,
            21.739 * D ** 0.580,
            c=self.obs_names["KMix1"],
            zorder=4,
            linewidth=self.linewidth,
            label="K2020 Mix1 [n=105k]",
        )

        # Karrer 2020
        D = np.arange(0.0001, 0.01, 0.0001)  # m
        # D = np.arange(0.01, 10.0, 0.0001)  # mm
        self.ax.plot(
            D * 1000,
            8.567 * D ** 0.393,
            c=self.obs_names["KMix2"],
            linewidth=self.linewidth,
            zorder=4,
            label="K2020 Mix2 [n=105k]",
        )

        # aggregates of unrimed sideplanes
        D = np.arange(0.5, 4.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.82 * D ** 0.12,
            c=self.obs_names["LH74 sideplane agg"],
            linewidth=self.linewidth,
            zorder=3,
            label="LH74 sideplane aggregates [n=23]",
        )

        # aggregates of unrimed radiating assemblages of
        # plates, sideplanes, bullets, and columns
        D = np.arange(0.2, 3.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.69 * D ** 0.41,
            c=self.obs_names["LH74 mix"],
            linewidth=self.linewidth,
            zorder=3,
            label="LH74 unrimed assemblage mix [n=31]",
        )

        # Locatelli and Hobbs 1974
        # aggregates of unrimed radiating assemblages of dendrites
        D = np.arange(2.0, 10.0, 0.01)  # mm
        self.ax.plot(
            D,
            0.8 * D ** 0.16,
            c=self.obs_names["LH74 dendrite"],
            linewidth=self.linewidth,
            zorder=3,
            label="LH74 unrimed assemblage dendrite [n=28]",
        )

        #         D = np.arange(0.4, 1.2, 0.01)  # mm
        #         self.ax.plot(
        #             D,
        #             0.81 * D * 0.99,
        #             c=self.colors_others[3],
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
        # self.ax.set_ylim(0.0, 1.0)
        self.ax.set_ylim(0.002, 10.0)
        self.ax.set_xlim([4e-2, 1e2])
        # self.ax.set_xlim(0.0, 10)
        self.ax.set_xscale("log")
        if mflag != "area":
            self.ax.set_xlabel("$D_{max}$ [mm]")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)

    def area_plot(self, title, xlabel, nm):
        """
        box plot for area and volumne of polygons from IPAS and CPI data
        plotted as a function of aspect ratio for IPAS
        CPI data threshold value included for reference
        """
        Vps, Ves, Vrs, Aps, Acs, Ars = [], [], [], [], [], []
        for self.phi_idx in self.phi_idxs:

            Ap = self.get_modes(self.Aps[self.phi_idx, self.r_idx, :, self.nm])
            Aps.append(Ap)
            if self.phi_idx == 1:
                print(Ap)

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
                "Ap IPAS": Aps,
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
        }
        self.ax = df.plot.bar(rot=0, color=color, ax=self.ax, legend=False, width=0.7)

        self.ax.hlines(
            Ap_CPI,
            xmin=-1.0,
            xmax=50.0,
            color="#DAF1DE",
            linewidth=self.linewidth,
            label="Ap_CPI",
        )
        self.ax.hlines(
            Ar_CPI,
            xmin=-1.0,
            xmax=50.0,
            linewidth=self.linewidth,
            color="#235347",
            label="Ar_CPI",
        )

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
            m_ellipsoid_vol = self.mass_ellipsoid_volumes()  #  kg
            m_ellipsoid_vol = self.get_modes(m_ellipsoid_vol)
            m_vols.append(m_ellipsoid_vol)

            m_spheroid_area = self.mass_spheroid_areas()  #  kg
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

    def Re_plot(self, title, xlabel):

        Re_areas, Re_vols = [], []
        for self.phi_idx in self.phi_idxs:

            if self.ASPECT_RATIOS[self.phi_idx] < 1.0:
                T = -15
            else:
                T = -5

            self.RHO_A = (
                1.2754 * (self.P / 1000) * (273.15 / (T + 273.15))
            )  # air density for a given pressure and temp
            self.dynamic_viscosity(T)

            # D = self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]

            # best number using area
            Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
            # Ap = self.Aps[self.phi_idx, self.r_idx, :, self.nm]
            m_spheroid_area = self.mass_spheroid_areas()
            # X_area = self.best_number_Mitchell(Ar, Ap, D, m_spheroid_area)  # shape of 300

            Xs = self.best_number_Heymsfield(Ar, m_spheroid_area)
            Res = self.reynolds_number_Heymsfield(Xs)
            Re_areas.append(self.get_modes(Res))

            # best number using volume
            Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
            # Vp = self.Vps[self.phi_idx, self.r_idx, :, self.nm]
            m_ellipsoid_vol = self.mass_ellipsoid_volumes()  #  kg

            Xs = self.best_number_Heymsfield(Vr, m_ellipsoid_vol)
            Res = self.reynolds_number_Heymsfield(Xs)
            Re_vols.append(self.get_modes(Res))

        df = pd.DataFrame(
            {"Re Area": Re_areas, "Re Volume": Re_vols}, index=self.ASPECT_RATIOS
        )
        color = {"Re Area": "#3d5a80", "Re Volume": "#E26610"}
        self.ax = df.plot.bar(rot=0, color=color, width=0.7, ax=self.ax, legend=False)

        self.ax.set_yscale("log")
        self.ax.grid(which="major")
        # self.ax.grid(which="minor")
        # self.ax.grid(True)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Reynolds Number", color="maroon")
        self.ax.set_title(title)

    def best_number_plot(self, title, xlabel):

        X_areas, X_vols = [], []
        for self.phi_idx in self.phi_idxs:

            if self.ASPECT_RATIOS[self.phi_idx] < 1.0:
                T = -15
            else:
                T = -5

            self.RHO_A = (
                1.2754 * (self.P / 1000) * (273.15 / (T + 273.15))
            )  # air density for a given pressure and temp
            self.dynamic_viscosity(T)

            D = self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]

            # best number using area
            Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
            Ap = self.Aps[self.phi_idx, self.r_idx, :, self.nm]
            m_spheroid_area = self.mass_spheroid_areas()
            X_area = self.best_number_Mitchell(
                Ar, Ap, D, m_spheroid_area
            )  # shape of 300

            X_area = self.get_modes(X_area)
            X_areas.append(X_area)

            # best number using volume
            Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
            Vp = self.Vps[self.phi_idx, self.r_idx, :, self.nm]
            m_ellipsoid_vol = self.mass_ellipsoid_volumes()  #  kg
            X_vol = self.best_number_Mitchell(Vr, Vp, D, m_ellipsoid_vol)
            X_vol = self.get_modes(X_vol)
            X_vols.append(X_vol)

        df = pd.DataFrame(
            {"X Area": X_areas, "X Volume": X_vols}, index=self.ASPECT_RATIOS
        )
        color = {"X Area": "#3d5a80", "X Volume": "#E26610"}
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
        self.ax.set_ylabel("Density [kg/$m^3$]", color="maroon")
        self.ax.set_title(title)

    def area_ratio_plot(self, title, xlabel):

        Ars, Vrs = [], []
        for self.phi_idx in self.phi_idxs:

            Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
            Vr = self.get_modes(Vr)
            Vrs.append(Vr)

            Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
            Ar = self.get_modes(Ar)
            Ars.append(Ar)

        df = pd.DataFrame({"Area": Ars, "Volume": Vrs}, index=self.ASPECT_RATIOS)
        color = {"Area": "#3d5a80", "Volume": "#E26610"}
        self.ax = df.plot.bar(rot=0, color=color, width=0.7, ax=self.ax, legend=False)

        df = self.df_CPI[self.df_CPI["classification"] == "agg"]
        df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
        Ar_CPI = df["area_ratio"]
        Ar_CPI = self.get_modes(Ar_CPI)

        self.ax.hlines(
            Ar_CPI,
            xmin=-1.0,
            xmax=50.0,
            linewidth=self.linewidth,
            color="#235347",
            label="CPI",
        )

        # self.ax.set_yscale('log')
        self.ax.grid(which="major")
        # self.ax.grid(which="minor")
        # self.ax.grid(True)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Area or Volume Ratio", color="maroon")
        self.ax.set_title(title)

    def aspect_ratios_cpi(self):

        for i, part_type in enumerate(self.particle_types):
            df = self.df_CPI[self.df_CPI["classification"] == part_type]
            self.ax.hist(df["phi"], color=self.colors_cpi[i])
