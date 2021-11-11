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

    def func(self, x, a, b):
        return np.log(a) + b * np.log(x)

    def curve_fit_plot(self, x, y):
        popt, pcov = curve_fit(self.func, x, y)
        fittedA = popt[0]
        fittedB = popt[1]
        return self.func(x, fittedA, fittedB)

    def curve_fit_CPI(self, x, y, deg=2):
        coefs = poly.polyfit(x, y, deg)
        ffit = poly.polyval(x, coefs)
        return ffit
        # error = np.std(modes_flat_ba[phi,::interval])
        # axs[0].fill_between(Ns, modes_flat_ba[phi,::interval]-error, modes_flat_ba[phi,::interval]+error, color=colors[phi], alpha =0.2)

    def m_D_plot(self, title, ylabel, mflag="vol"):

        colors = ["#E0B069", "#B55F56", "#514F51", "#165E6E", "#A0B1BC"]

        #        D_modes = np.zeros((len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3]))
        #         m_area_modes = np.zeros(
        #             (len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3])
        #         )
        #         m_vol_modes = np.zeros(
        #             (len(self.phi_idxs), len(self.r_idxs), self.agg_as.shape[3])
        #         )

        for self.phi_idx in self.phi_idxs:
            # for nm in range(self.agg_as.shape[3]):
            # self.nm = nm
            #                     D_modes[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
            #                         self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
            #                     )
            #                     m_spheroid_area = self.mass_spheroid_areas(
            #                     )s
            #                     m_area_modes[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
            #                         m_spheroid_area
            #                     )

            #                     m_vol_modes[self.phi_idx, self.r_idx, self.nm] = self.get_modes(
            #                         self.mass_ellipsoid_volumes()
            #                     )  # kg

            df = pd.DataFrame(
                {
                    "D": self.Dmaxs[self.phi_idx, :, :, :].flatten() * 1000,
                    "m": self.mass_spheroid_areas(),
                }
            )

            bins = np.logspace(-1.5, 2.1, 20)
            df["D_ranges"] = pd.cut(df["D"], bins=bins)

            # df.dropna(inplace=True)
            df.boxplot(
                column="m",
                by="D_ranges",
                ax=self.ax,
                showfliers=False,
                color=colors[self.phi_idx],
            )

        #                     if mflag != 'area':
        # #                         ### IPAS ###
        #                         sc_IPAS = self.ax.scatter(
        #                             D_modes[self.phi_idx, self.r_idx, self.nm] * 1000,
        #                             m_vol_modes[self.phi_idx, self.r_idx, self.nm],
        #                             s=self.nm,
        #                             c=colors[self.phi_idx],
        #                         )

        # make sure D modes are increasing to plot between datapoints:
        #         D_inc = []
        #         for self.phi_idx in self.phi_idxs:
        #             starting_D = D_modes[self.phi_idx, 0, 0]
        #             for nm in range(1, self.agg_as.shape[3]-1):
        #                 if nm == 0:
        #                     if self.D_modes[self.phi_idx, 0, nm+] > starting_D:
        #                         D_inc.append(nm)
        #                 else:
        #                     if self.D_modes[self.phi_idx, 0, nm+] > self.D_modes[self.phi_idx, 0, nm]

        #         m = m_area_modes if mflag == 'area' else m_vol_modes
        #         D_modes = D_modes * 1000  # converting to mm for figure
        #         for self.phi_idx in self.phi_idxs:
        #             for self.r_idx in self.r_idxs:

        #                 if mflag == 'area':
        #                     ax = sns.regplot(x = D_modes[self.phi_idx,self.r_idx, :],
        #                                y = m[self.phi_idx, self.r_idx, :],
        #                                ci=None,
        #                                ax=self.ax,
        #                                scatter= True,
        #                                color = colors[self.phi_idx],
        #                                label = self.ASPECT_RATIOS[self.phi_idx] if self.r_idx == 0 else "")

        # plot curve fits instead of scatter to minimize the 'business' of the plot
        # y = self.curve_fit_plot(D_modes[self.phi_idx,self.r_idx, :], m[self.phi_idx, self.r_idx, :])
        # self.ax.plot(D_modes[self.phi_idx,self.r_idx, 7:], y[7:], color=colors[self.phi_idx], linewidth=3)
        # self.ax.legend(title="Monomer\nAspect Ratio", loc="lower right")

        # CPI
        df = self.df_CPI[self.df_CPI["classification"] == "agg"]
        # y = self.curve_fit_CPI(df['a']*1000, self.mass_CPI(df))
        # self.ax.plot(df['a']*1000, y, linewidth=3, c='g')

        # sns.regplot(x=df['a']*1000, y=self.mass_CPI(df), scatter=False, color ='b', ax=self.ax)

        #         self.ax.scatter(
        #             df['a']*1000,
        #             self.mass_CPI(df),
        #             linestyle='--',
        #             c="g",
        #             label="CPI agg",
        #         )

        #         df = self.df_CPI[self.df_CPI['classification'] == 'column']
        #         self.ax.scatter(
        #             df['a']*1000,
        #             self.mass_CPI(df),
        #             linestyle='--',
        #             c="darkgreen",
        #             label="CPI column",
        #         )

        #         df = self.df_CPI[self.df_CPI['classification'] == 'compact_irreg']
        #         print(len(df))
        #         self.ax.scatter(
        #             df['a']*1000,
        #             self.mass_CPI(df),
        #             linestyle='--',
        #             c="darkgreen",
        #             label="CPI column",
        #         )

        ### MITCHELL 1996 ###
        # crystal with sector-like branches
        # 10 mu <= D <= 40 mu
        #         D = np.arange(1E-5, 4E-5, 0.0001)
        #         m_sector = 0.00614*D**2.42
        #         plt.plot(
        #             D,
        #             m_sector,
        #             c="cyan",
        #             label="Mitchel (1996) small sectors",
        #         )

        #         # crystal with sector-like branches
        #         # 40 mu < D <= 2000 mu
        #         D = np.arange(4E-5, 0.002, 0.0001)
        #         m_sector = 0.00142*D**2.02
        #         plt.plot(
        #             D,
        #             m_sector,
        #             c="gray",
        #             label="Mitchel (1996) large sectors",
        #         )

        # aggregates of side planes, columns, and bullets
        # 800 mu <= D <= 4500 mu
        #         D = np.arange(800, 4500, 0.0001)  # mu
        #         m_aggs = 0.0028 * D ** 2.1
        #         self.ax.plot(D*0.001, m_aggs*0.001, c="k", linewidth=3, label="M96 aggregates ")

        #  Mitchell 1990
        #  aggregates of side planes, bullets, and columns
        D = np.arange(0.8, 4.5, 0.0001)  # mm
        m_aggs = (0.022 * D ** 2.1) * 1e-6
        self.ax.plot(
            D,
            m_aggs,
            linestyle="--",
            c="lightgoldenrodyellow",
            linewidth=3,
            label="M90 column aggregates",
        )

        #  aggregates of radiating assemblages of plates
        D = np.arange(0.8, 7.7, 0.0001)  # mm
        m_aggs = (0.023 * D ** 1.8) * 1e-6
        self.ax.plot(
            D,
            m_aggs,
            c="darkolivegreen",
            linewidth=3,
            linestyle="--",
            label="M90 plate aggregates",
        )

        ### Locatellii and Hobbs 1974 ###
        D = np.arange(1.0, 3.0, 0.0001)  # mm
        m = (0.037 * D ** 1.9) * 1e-6  # mg
        self.ax.plot(
            D,
            m,
            c="darkred",
            linewidth=3,
            linestyle="--",
            label="LH74 mixed aggregates",
        )

        D = np.arange(2, 10, 0.0001)  # mm
        m = (0.073 * D ** 1.4) * 1e-6
        self.ax.plot(
            D,
            m,
            c="darkslateblue",
            linewidth=3,
            linestyle="--",
            label="LH74 dendritic aggregates",
        )

        #         legend1 = self.ax.legend(
        #             *sc_IPAS.legend_elements("sizes", num=6),
        #             loc="lower right",
        #             title="number of\nmonomers",
        #         )
        #         self.ax.add_artist(legend1)

        #         legend2 = self.ax.legend(
        #             *sc_IPAS.legend_elements("colors", num=6),
        #             loc="upper left",
        #             title="monomer\naspect ratio",
        #         )
        #         self.ax.add_artist(legend2)

        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)
        self.ax.set_yscale("log")
        self.ax.set_xscale("log")
        self.ax.set_xlabel("$D_{max}$ [mm]")
        self.ax.set_ylabel(ylabel)
        # self.ax.set_ylim([1e-12, 2e-1])
        # self.ax.set_xlim([5e-1, 2e1])
        self.ax.set_title(title)
        # self.ax.set_xticks(df['D_ranges'].unique())

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

        #         legend1 = self.ax.legend(
        #             *vt_IPAS.legend_elements("colors", num=5),
        #             loc="lower right",
        #             title="   monomer\naspect ratio",
        #         )
        #         self.ax.add_artist(legend1)

        #         lines = plt.gca().get_lines()
        #         #include = [0,1]
        #         #legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include], loc=1)
        #         legend1 = plt.legend([lines[i] for i in [2,3,4]],['LH1','LH2'], loc=4)
        #         plt.gca().add_artist(legend1)

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
