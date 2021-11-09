import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mD_vT_relationships as relationships
import numpy as np
import pandas as pd


class Plots(relationships.Relationships):
    """
    mass-dimensional and terminal velocity plotting code
    """

    def __init__(
        self, ax, agg_as, agg_bs, agg_cs, phi_idxs, r_idxs, Aps, Acs, Vps, Ves, Dmaxs
    ):
        super().__init__(
            agg_as, agg_bs, agg_cs, phi_idxs, r_idxs, Aps, Acs, Vps, Ves, Dmaxs
        )
        self.ax = ax

    def m_D_plot(self):
        colors_vol = ["gold", "#efc070", "#e47025", "darkred"]
        colors_area = ["#689689", "#31698a", "#1c3464", "#432865"]

        for self.r_idx in self.r_idxs:
            for self.phi_idx in self.phi_idxs:

                m_ellipsoid_vol = self.mass_ellipsoid_volumes()
                m_ellipsoid_vol = self.get_modes(m_ellipsoid_vol)

                Ar = self.get_modes(Ar[self.phi_idx, self.r_idx, :, :])

                m_spheroid_area = self.mass_spheroid_areas(Ar)
                m_spheroid_area = self.get_modes(m_spheroid_area)

                D = self.Dmaxs[self.phi_idx, self.r_idx, :, :]
                D = self.get_modes(D)

                ### COX 1988 ###
                m_cox = 0.06 * D ** 2.07
                plt.plot(D, m_cox, c="k", linewidth=3, label="Cox (1988)")

                ### IPAS ###
                plt.scatter(
                    D,
                    m_ellipsoid_vol,
                    s=self.nm,
                    c=colors_vol[self.phi_idx],
                    label="IPAS (Volume)",
                )

                sc_IPAS = plt.scatter(
                    D,
                    m_spheroid_area,
                    s=self.nm,
                    c=colors_area[self.phi_idx],
                    label="IPAS (Area)",
                )

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
        D = np.arange(4e-5, 0.002, 0.0001)
        m_aggs = 0.0028 * D ** 2.1
        plt.plot(D, m_aggs, c="darkgreen", linewidth=3, label="M96 aggregates ")

        #  Mitchell 1990
        #  aggregates of side planes, bullets, and columns
        D = np.arange(0.0008, 0.0045, 0.0001)
        m_aggs = 0.022 * D ** 2.1
        plt.plot(D, m_aggs, c="limegreen", linewidth=3, label="M90 column aggregates")

        #  Mitchell 1990
        #  aggregates of radiating assemblages of plates
        D = np.arange(0.0008, 0.0077, 0.0001)
        m_aggs = 0.023 * D ** 1.8
        plt.plot(D, m_aggs, c="lightgreen", linewidth=3, label="M90 plate aggregates")

        ### Locatellii and Hobbs 1974 ###
        D = np.arange(0.001, 0.003, 0.0001)
        m = 0.037 * D ** 1.9

        plt.plot(D, m, c="red", linewidth=3, label="LH74 mixed aggregates")

        D = np.arange(0.002, 0.01, 0.0001)
        m = 0.073 * D ** 1.4

        plt.plot(D, m, c="gold", linewidth=3, label="LH74 dendritic aggregates")

        legend1 = plt.legend(
            *sc_IPAS.legend_elements("sizes", num=6),
            loc="lower right",
            title="number of\nmonomers",
        )
        self.ax.add_artist(legend1)

        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)
        self.ax.set_yscale("log")
        self.ax.set_xscale("log")
        self.ax.set_xlabel("$D_{max}$ [m]")
        self.ax.set_ylabel("m [kg]")
        self.ax.set_ylim([1e-12, 2e-1])
        self.ax.set_xlim([5e-5, 1e-1])

        self.ax.set_title("Random Orientation")
        # self.ax.set_title('Quasi-Horizontal Orientation')

    def vt_plot(self):

        colors = ["#E0B069", "#B55F56", "#514F51", "#165E6E", "#A0B1BC"]

        for self.phi_idx in self.phi_idxs:
            for self.r_idx in self.r_idxs:
                for nm in range(99):
                    self.nm = nm
                    # print(self.phi_idx, self.nm)

                    self.dynamic_viscosity()
                    D = self.Dmaxs[self.phi_idx, self.r_idx, :, self.nm]
                    D = self.get_modes(D)

                    # best number using area
                    Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]
                    Ap = self.Aps[self.phi_idx, self.r_idx, :, self.nm]
                    Ac = self.Acs[self.phi_idx, self.r_idx, :, self.nm]
                    #                     m_spheroid_area = self.mass_spheroid_areas(Ar)
                    #                     X_area = self.best_number(Ar, D, Ap, Ac, m_spheroid_area)  # shape of 300
                    #                     X_area = self.get_modes(X_area)
                    #                     Re_area = self.reynolds_number(X_area)

                    # best number using volume
                    Vr = self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
                    # Vp = self.Vps[self.phi_idx, self.r_idx, :, self.nm]
                    m_ellipsoid_vol = self.mass_ellipsoid_volumes(Vr)  #  kg
                    X_vol = self.best_number(Ar, D, Ap, Ac, Ar, m_ellipsoid_vol)
                    X_vol = self.get_modes(X_vol)
                    Re_vol = self.reynolds_number(X_vol)
                    # X = self.best_number_Mitchell(Ar)

                    # self.terminal_velocity_Mitchell_2005(Ar, X)
                    self.terminal_velocity_Mitchell(D, Re_vol)
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
                    vt_IPAS = plt.scatter(
                        D,
                        self.vt,
                        c=colors[self.phi_idx],
                        label=self.ASPECT_RATIOS[self.phi_idx],
                    )

        # Locatelli and Hobbs 1974
        # aggregates of unrimed radiating assemblages of dendrites
        D = np.arange(2.0, 10.0, 0.01)
        plt.plot(
            D,
            0.8 * D * 0.16,
            c="olivedrab",
            linewidth=3,
            label="LH unrimed assemblage dendrite",
        )

        # aggregates of unrimed radiating assemblages of
        # plates, sideplanes, bullets, and columns
        D = np.arange(0.2, 3.0, 0.01)
        plt.plot(
            D,
            0.069 * D * 0.41,
            c="darkolivegreen",
            linewidth=3,
            label="LH unrimed assemblage mix",
        )

        # aggregates of unrimed sideplanes
        D = np.arange(0.5, 4.0, 0.01)
        plt.plot(
            D, 0.082 * D * 0.12, c="khaki", linewidth=3, label="LH unrimed sideplanes"
        )

        legend1 = plt.legend(
            *vt_IPAS.legend_elements("sizes", num=6),
            loc="lower right",
            title="number of\nmonomers",
        )
        self.ax.add_artist(legend1)

        #         lines = plt.gca().get_lines()
        #         #include = [0,1]
        #         #legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include], loc=1)
        #         legend1 = plt.legend([lines[i] for i in [2,3,4]],['LH1','LH2'], loc=4)
        #         plt.gca().add_artist(legend1)

        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)
        self.ax.set_yscale("log")
        self.ax.set_ylim(0.01, 5.0)
        # self.ax.set_xlim(0.05, 30)
        # self.ax.set_xlim(0.0, 10)
        self.ax.set_xscale("log")
        self.ax.set_xlabel("$D_{max}$ [mm]")
        self.ax.set_ylabel("$V_t$ [m/s]")

        self.ax.set_title("Random Orientation")
        # self.ax.set_title("Quasi-Horizontal Orientation")

    def area_plot(self, title, xlabel, nm):

        Vps, Ves, Vrs, Aps, Acs, Ars = [], [], [], [], [], []
        for self.phi_idx in self.phi_idxs:

            Ar = self.get_modes(self.Ars[self.phi_idx, self.r_idx, :, self.nm])
            Ars.append(Ar)
            Ap = self.get_modes(self.Aps[self.phi_idx, self.r_idx, :, self.nm])
            Aps.append(Ap)
            Ac = self.get_modes(self.Acs[self.phi_idx, self.r_idx, :, self.nm])
            Acs.append(Ac)

            Vr = self.get_modes(self.Vrs[self.phi_idx, self.r_idx, :, self.nm])
            Vrs.append(Vr)
            Vp = self.Vps[self.phi_idx, self.r_idx, :, self.nm]
            Vps.append(Vp)

            Ve = self.get_modes(self.Ves[self.phi_idx, self.r_idx, :, self.nm])
            Ves.append(Ve)

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
        self.ax = df.plot.bar(rot=0, color=color, width=0.75, ax=self.ax, legend=False)

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
