import warnings

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class Relationships:
    """
    mass-dimensional and terminal velocity plotting code
    """

    ASPECT_RATIOS = [0.01, 0.1, 1.0, 10.0, 50.0]
    RHO_A = 1.395  # air density at -20C kg/m^3
    RHO_B = 916.8  # bulk density of ice [kg/m^3]
    GRAVITY = 9.81  # [m/sec^2]

    def __init__(
        self, agg_as, agg_bs, agg_cs, phi_idxs, r_idxs, Aps, Acs, Vps, Ves, Dmaxs
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
        self.Ars = Aps / Acs
        self.Vps = Vps * 1e-18  # [m3]
        self.Ves = Ves * 1e-18  # [m3]
        self.Vrs = Vps / Ves
        self.Dmaxs = Dmaxs * 1e-6  # [m]
        self.nms = np.arange(0, 99, 1)  # number of monomers

    def get_modes(self, var):
        hist, bin_edges = np.histogram(var, bins=30)
        mode_index = hist.argmax()
        # plt.hist(var, bins=30)
        # plt.show()
        return bin_edges[mode_index]

    def mass_CPI(self, df):
        # a the longer axis always

        rho_i = self.RHO_B * df["area_ratio"]

        m_spheroid = 4 / 3 * np.pi * df["a"] ** 2 * df["c"] * rho_i  # kg
        return m_spheroid

    def mass_ellipsoid_volumes(self):

        m_ellipsoid = (
            4
            / 3
            * np.pi
            * self.agg_as[self.phi_idx, self.r_idx, :, self.nm]
            * self.agg_bs[self.phi_idx, self.r_idx, :, self.nm]
            * self.agg_cs[self.phi_idx, self.r_idx, :, self.nm]
            * self.RHO_B
            * self.Vrs[self.phi_idx, self.r_idx, :, self.nm]
        )  # kg

        return m_ellipsoid

    def mass_spheroid_areas(self):
        agg_as = self.agg_as[self.phi_idx, self.r_idx, :, self.nm]
        agg_bs = self.agg_bs[self.phi_idx, self.r_idx, :, self.nm]
        agg_cs = self.agg_cs[self.phi_idx, self.r_idx, :, self.nm]
        Ar = self.Ars[self.phi_idx, self.r_idx, :, self.nm]

        m_spheroid = np.zeros((len(agg_as)))
        for n in range(len(agg_as)):
            rho_i = self.RHO_B * Ar[n]
            if (agg_bs[n] - agg_cs[n]) <= (agg_as[n] - agg_bs[n]):
                # prolate
                m_spheroid[n] = 4 / 3 * np.pi * agg_as[n] * agg_cs[n] ** 2 * rho_i  # kg
            else:
                # oblate
                m_spheroid[n] = 4 / 3 * np.pi * agg_as[n] ** 2 * agg_cs[n] * rho_i  # kg
        return m_spheroid

    def best_number_Mitchell(self, Ar, Ap, D, m):

        rho_p = self.RHO_B * (Ar)
        X = (
            (2 * m / rho_p)
            * ((rho_p - self.RHO_A) * self.GRAVITY * self.RHO_A / self.eta ** 2)
            * (D ** 2 / Ap)
            * (Ar) ** (3 / 4)
        )
        # print(rho_p)
        # print('Ap, X, m', Ap, X, m)
        return X

    def best_number_Heymsfield(self, Ar, m):

        X = (self.RHO_A / self.eta ** 2) * (
            8 * m * self.GRAVITY / (np.pi * np.sqrt(Ar))
        )
        return X

    def reynolds_number_Mitchell(self, X):

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
        if X > 1.0e8:
            # print('bad')
            a = 1.0
            b = 0.4
        # return 8.5 * ((1 + 0.1519 * X ** (1 / 2)) ** (1 / 2) - 1) ** 2
        # print('a, x, b', a * X ** b)
        return a * X ** b

    def reynolds_number_Heymsfield(self, X):
        Co = 0.35
        delta_o = 8.0
        Re = (
            delta_o ** 2
            / 4
            * (((1 + (4 * np.sqrt(X)) / (delta_o ** 2 * np.sqrt(Co))) ** (1 / 2)) - 1)
            ** 2
        )
        return Re

    def dynamic_viscosity(self):
        # only true with T < 0C
        eta = 1.718 + 0.0049 * self.T - 1.2e-5 * self.T ** 2
        self.eta = eta * 1e-4 * (100 / 1000)

    def kinematic_viscosity(self):
        # return (0.000001458 * self.T ** (3 / 2)) / (self.T + 110.4)
        return self.eta / self.RHO_A
        # return 1.81E-5
        # should be around 1.17E-5 m2/s

    def terminal_velocity(self, D, Re):
        self.vt = (self.eta * Re) / (self.RHO_A * D)
