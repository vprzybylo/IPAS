import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class Plots:
    """
    mass-dimensional and terminal velocity plotting code
    """

    RHO_A = 1.395  # air density at -20C kg/m^3
    RHO_B = 916.8  # bulk density of ice [kg/m^3]
    GRAVITY = 9.81  # [m/sec^2]

    def __init__(
        self, ax, agg_as, agg_bs, agg_cs, phi_idxs, r_idxs, Aps, Acs, Vps, Ves, Dmaxs
    ):
        """
        args (all arrays are of [mono phi, mono r, nclusters, ncrystals]):
        -----
        agg_as (array)= aggregate major radius from fit ellipsoid
        agg_bs (array)= aggregate middle radius from fit ellipsoid
        agg_cs (array)= aggregate minor radius from fit ellipsoid
        phi_idx (int)= monomer aspect ratio index
        r_idx (int)= monomer radius index
        Aps(array)= area of projected aggregate polygons in x-y plane
        Acs (array)= area of smallest circle fit around projected aggregate in x-y plane
        Vps (array)= volume of aggregate polygons
        Ves (array)= volume of ellipsoid
        Dmaxs (array) = longest axis from vertex to vertex through 3D polygon
        """
        self.agg_as = agg_as * 1e-6  # [m]
        self.agg_bs = agg_bs * 1e-6
        self.agg_cs = agg_cs * 1e-6
        self.phi_idxs = phi_idxs
        self.r_idxs = r_idxs
        self.Aps = Aps * 1e-12  # [m2]
        self.Acs = Acs * 1e-12  # [m2]
        self.Vps = Vps * 1e-18  # [m3]
        self.Ves = Ves * 1e-18  # [m3]
        self.Dmaxs = Dmaxs * 1e-6  # [m]
        self.nm = np.arange(0, 99, 1)  # number of monomers
        self.ax = ax

    def get_modes(self, var):
        """return mode of variable across all aggregates for a given monomer phi and r"""
        modes = np.zeros(shape=(var.shape[1]))
        for nm in range(var.shape[1]):
            hist, bin_edges = np.histogram(var[:, nm], bins=30)
            mode_index = hist.argmax()
            modes[nm] = bin_edges[mode_index]
        return modes

    def mass_ellipsoid_volumes(self):
        rho_i = self.RHO_B * (
            self.Vps[self.phi_idx, self.r_idx, :, :]
            / self.Ves[self.phi_idx, self.r_idx, :, :]
        )

        m_spheroid = (
            4
            / 3
            * np.pi
            * self.agg_as[self.phi_idx, self.r_idx, :, :]
            * self.agg_bs[self.phi_idx, self.r_idx, :, :]
            * self.agg_cs[self.phi_idx, self.r_idx, :, :]
            * rho_i
        )  # kg

        # print('m spheroid', m_spheroid)
        return m_spheroid

    def mass_spheroid_areas(self):

        rho_i = self.RHO_B * (
            self.Aps[self.phi_idx, self.r_idx, :, :]
            / self.Acs[self.phi_idx, self.r_idx, :, :]
        )

        m_spheroid = (
            4
            / 3
            * np.pi
            * self.agg_as[self.phi_idx, self.r_idx, :, :] ** 2
            * self.agg_cs[self.phi_idx, self.r_idx, :, :]
            * rho_i
        )  # kg
        return m_spheroid

    def m_D_plot(self):
        colors_vol = ["gold", "#efc070", "#e47025", "darkred"]
        colors_area = ["#689689", "#31698a", "#1c3464", "#432865"]

        for self.r_idx in self.r_idxs:
            for self.phi_idx in self.phi_idxs:

                m_ellipsoid_vol = self.mass_ellipsoid_volumes()
                m_ellipsoid_vol = self.get_modes(m_ellipsoid_vol)

                m_spheroid_area = self.mass_spheroid_areas()
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

    def terminal_velocity(self, T):

        m_ellipsoid_vol = self.mass_ellipsoid_volumes()  #  kg
        m_ellipsoid_vol = self.get_modes(m_ellipsoid_vol)

        rho_a = 1.225  # kg/m^3
        eta = 1.718 + 0.0049 * T - 1.2e-5 * T ** 2
        eta = eta * 1e-4 * (100 / 1000)
        # eta = 1.63E10-5  # kg/ms
        print(eta)
        area_ratio = (
            self.Acs[self.phi_idx, self.r_idx, :, :]
            / self.Aps[self.phi_idx, self.r_idx, :, :]
        )

        X = (
            (8 * m_ellipsoid_vol * self.GRAVITY * rho_a) / (np.pi * eta ** 2)
        ) * area_ratio ** (1 / 4)
        Re = 8.5 * ((1 + 0.1519 * X ** (1 / 2)) ** (1 / 2) - 1) ** 2

        # print(Re)
        self.vt = (eta * Re / (2 * rho_a)) * (
            np.pi / self.Aps[self.phi_idx, self.r_idx, :, :]
        ) ** (1 / 2)

    def best_number(self, eta):
        rho_p = self.RHO_B * (
            self.Aps[self.phi_idx, self.r_idx, :, :]
            / self.Acs[self.phi_idx, self.r_idx, :, :]
        )
        # print('rho_p', rho_p)

        m_ellipsoid_vol = self.mass_ellipsoid_volumes()  #  kg
        m_ellipsoid_vol = self.get_modes(m_ellipsoid_vol)

        X = (
            (2 * m_ellipsoid_vol / rho_p)
            * ((rho_p - self.RHO_A) * self.GRAVITY * self.RHO_A / eta ** 2)
            * (
                self.Dmaxs[self.phi_idx, self.r_idx, :, :] ** 2
                / self.Aps[self.phi_idx, self.r_idx, :, :]
            )
        )

        return X

    def reynolds_number(self, Xs):

        # X has a shape of 99 for all number of monomers
        # loop through each to calculate Re for each index
        Res = []
        for X in Xs:
            if X <= 10:
                a = 0.04394
                b = 0.970
            if X > 10 and X <= 585:
                a = 0.06049
                b = 0.831
            if X > 585 and X <= 1.56e5:
                a = 0.2072
                b = 0.638
            if X > 1.56e5 and X < 1.0e8:
                a = 1.0865
                b = 0.499
            Res.append(a * X ** b)
        # print('Res', min(Res), max(Res))
        return Res

    def eta(self, T):
        eta = 1.718 + 0.0049 * T - 1.2e-5 * T ** 2
        return eta * 1e-4 * (100 / 1000)

    def terminal_velocity_Mitchell(self, T, D, Re, eta):
        self.vt = (np.array([eta] * len(Re)) * Re) / (self.RHO_A * D)

    def vt_plot(self, T=-15):

        colors = ["#E0B069", "#B55F56", "#514F51", "#165E6E", "#A0B1BC"]
        aspect_ratios = [0.01, 0.1, 1.0, 10.0, 50.0]
        for self.phi_idx in self.phi_idxs:
            for self.r_idx in self.r_idxs:
                eta = self.eta(T)
                D = self.Dmaxs[self.phi_idx, self.r_idx, :, :]
                D = self.get_modes(D)

                X = self.best_number(eta)
                X = self.get_modes(X)

                Re = self.reynolds_number(X)
                self.terminal_velocity_Mitchell(T, D, Re, eta)
                # self.terminal_velocity(T)

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
                # print(D)
                vt_IPAS = plt.scatter(
                    D,
                    self.vt,
                    s=self.nm,
                    c=colors[self.phi_idx],
                    linewidth=3,
                    label=aspect_ratios[self.phi_idx],
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
                    D,
                    0.082 * D * 0.12,
                    c="khaki",
                    linewidth=3,
                    label="LH unrimed sideplanes",
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
        self.ax.set_xlim(0.05, 30)
        # self.ax.set_xlim(0.0, 10)
        self.ax.set_xscale("log")
        self.ax.set_xlabel("$D_{max}$ [mm]")
        self.ax.set_ylabel("$V_t$ [m/s]")

        self.ax.set_title("Random Orientation")
        # self.ax.set_title("Quasi-Horizontal Orientation")
