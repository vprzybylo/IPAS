import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class Plots:
    """
    mass-dimensional and terminal velocity plotting code
    """

    RHO_B = 916.8  # bulk density of ice [kg/m3]

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
        self.agg_as = agg_as
        self.agg_bs = agg_bs
        self.agg_cs = agg_cs
        self.phi_idxs = phi_idxs
        self.r_idxs = r_idxs
        self.Aps = Aps
        self.Acs = Acs
        self.Vps = Vps
        self.Ves = Ves
        self.Dmaxs = Dmaxs
        self.nm = np.arange(0, 99, 1)  # number of monomers
        self.ax = ax

    def convert_to_m(self, var):
        """ convert axes from micrometers to meters"""
        return var * 1e-6

    def get_modes(self, var):
        """return mode of variable across all aggregates for a given monomer phi and r"""
        modes = np.zeros(shape=(var.shape[0], var.shape[1], var.shape[3]))
        for phi in range(var.shape[0]):
            for r in range(var.shape[1]):
                for nm in range(var.shape[3]):
                    hist, bin_edges = np.histogram(var[phi, r, :, nm], bins=30)
                    mode_index = hist.argmax()
                    modes[phi, r, nm] = bin_edges[mode_index]
        return modes

    def mass_spheroid_volumes(self):

        rho_i = self.RHO_B * (
            self.Vps[self.phi_idx, self.r_idx, :, :]
            / self.Ves[self.phi_idx, self.r_idx, :, :]
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
        colors_vol = ["#e5e5f3", "#b2b2dc", "#3232a2", "#00008b"]
        colors_area = ["#ffd17a", "#ffbd42", "#ffab0f", "#ffa805"]

        for self.r_idx in self.r_idxs:
            for self.phi_idx in self.phi_idxs:
                m_spheroid_vol = self.mass_spheroid_volumes()
                m_spheroid_area = self.mass_spheroid_areas()

                D = self.Dmaxs[self.phi_idx, self.r_idx, :, :]

                m_cox = 0.06 * D ** 2.07
                plt.plot(
                    np.mean(D, axis=0),
                    np.mean(m_cox, axis=0),
                    c="g",
                    label="Cox (1988)",
                )
                m_large = 0.0257 * D ** 2.0
                plt.plot(
                    np.mean(D, axis=0),
                    np.mean(m_large, axis=0),
                    c="k",
                    label="Cox large xtals (1988)",
                )

                plt.scatter(
                    np.mean(D, axis=0),
                    np.mean(m_spheroid_vol, axis=0),
                    s=self.nm,
                    c=colors_vol[self.phi_idx],
                    label="IPAS (Volume)",
                )

                sc_IPAS = plt.scatter(
                    np.mean(D, axis=0),
                    np.mean(m_spheroid_area, axis=0),
                    s=self.nm,
                    c=colors_area[self.phi_idx],
                    label="IPAS (Area)",
                )

        #         self.ax.set_ylim([1e-8, 2e-5])
        #         self.ax.set_xlim([1e-4, 1e-2])
        legend1 = plt.legend(
            *sc_IPAS.legend_elements("sizes", num=6),
            loc="lower right",
            title="number of\nmonomers"
        )
        self.ax.add_artist(legend1)

        self.ax.grid(which="major")
        self.ax.grid(which="minor")
        self.ax.grid(True)
        self.ax.set_yscale("log")
        self.ax.set_xscale("log")
        self.ax.set_xlabel("$D_{max}$ [m]")
        self.ax.set_ylabel("m [kg]")
        self.ax.set_title("Random Orientation")
