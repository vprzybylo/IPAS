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
    T = -15  # temperature [C]
    P = 800  # pressure [hPa]
    RHO_A = (
        1.2754 * (P / 1000) * (273.15 / (T + 273.15))
    )  # air density for a given pressure and temp
    RHO_B = 916.8  # bulk density of ice [kg/m^3]
    GRAVITY = 9.81  # [m/sec^2]

    # Best # to Reynolds # power law fit coeffs
    ao = 1.0e-5
    bo = 1.0
    # sfc roughness
    delta_o = 5.83
    Co = 0.6
    C1 = 4 / (delta_o ** 2 * Co ** (1 / 2))
    C2 = delta_o ** 2 / 4
    # effective density coefficients
    k = (
        0.07
    )  # Exponent  in  the  terminal  velocity  versusdiameter relationship; aggs at ground H11 in chart
    n = 1.5  # Exponent in effective density relationship
    # Mitchell power laws for best number
    # all monomers and for limited size ranges
    # trying bullet rosette between D 200microns and 1000microns
    alpha = 0.00308
    beta = 2.26
    sigma = 1.57
    gamma = 0.0869

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
        #         plt.hist(var, bins=30)
        #         plt.show()
        return bin_edges[mode_index]

    def mass_CPI(self, df):
        # a the longer axis always
        m_spheroid = (
            4 / 3 * np.pi * df["a"] ** 2 * df["c"] * self.RHO_B * df["area_ratio"]
        )  # kg
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
        agg_as = self.agg_as[self.phi_idx, :, :, :]
        agg_bs = self.agg_bs[self.phi_idx, :, :, :]
        agg_cs = self.agg_cs[self.phi_idx, :, :, :]
        Ar = self.Ars[self.phi_idx, :, :, :]

        m_spheroid = np.zeros(
            (self.agg_as.shape[1], self.agg_as.shape[2], self.agg_as.shape[3])
        )
        for r in range(self.agg_as.shape[1]):
            for n in range(self.agg_as.shape[2]):
                for nm in range(self.agg_as.shape[3]):
                    rho_i = self.RHO_B * Ar[r, n, nm]

                    if (agg_bs[r, n, nm] - agg_cs[r, n, nm]) <= (
                        agg_as[r, n, nm] - agg_bs[r, n, nm]
                    ):
                        # prolate
                        m_spheroid[r, n, nm] = (
                            4
                            / 3
                            * np.pi
                            * agg_as[r, n, nm]
                            * agg_cs[r, n, nm] ** 2
                            * rho_i
                        )  # kg
                    else:
                        # oblate
                        m_spheroid[r, n, nm] = (
                            4
                            / 3
                            * np.pi
                            * agg_as[r, n, nm] ** 2
                            * agg_cs[r, n, nm]
                            * rho_i
                        )  # kg
        return m_spheroid.flatten()

    def b1(self, X):
        return (self.C1 * X ** (1 / 2)) / (
            2
            * ((1 + self.C1 * X ** (1 / 2)) ** (1 / 2) - 1)
            * (1 + self.C1 * X ** (1 / 2)) ** (1 / 2)
        ) - (
            self.ao
            * self.bo
            * X ** self.bo
            / (self.C2 * ((1 + self.C1 * X ** (1 / 2)) ** (1 / 2) - 1) ** 2)
        )

    def a1(self, X):
        return self.C2 * (
            (1 + self.C1 * X ** (1 / 2)) ** (1 / 2) - 1
        ) ** 2 - self.ao * X ** self.bo / (X ** self.b1(X))

    def terminal_velocity_Mitchell_2005(self, Ar, X):

        m_ellipsoid_vol = self.mass_ellipsoid_volumes()  #  kg
        m_ellipsoid_vol = self.get_modes(m_ellipsoid_vol)

        D = self.Dmaxs[self.phi_idx, self.r_idx, :, :]
        D = self.get_modes(D)

        u = self.kinematic_viscosity()

        self.vt = (
            self.a1(X)
            * ((4 * self.GRAVITY * self.k) / (3 * self.RHO_A)) ** self.b1(X)
            * u ** (1 - 2 * self.b1(X))
            * D ** (3 * self.b1(X) - 1)
            * Ar ** ((self.n - 1) * self.b1(X))
        )

        #     def best_number(self, Ar, D, Ap, Ac, m):

        #         rho_p = self.RHO_B * (Ar)
        #         X = (
        #             (2 * m / self.RHO_A)
        #             * ((rho_p - self.RHO_A) * self.GRAVITY * self.RHO_A / self.eta ** 2)
        #             * (D ** 2 / Ac)*(Ap/Ac)**(1/4)
        #         )
        #         #print(rho_p)
        #         #print('X, m, D, Ap', X, m, D, Ap)
        return X

    def best_number(self, Ar, D, Ap, Ac, qe, m):

        rho_p = self.RHO_B * (Ar)
        X = (
            (2 * m / rho_p)
            * ((rho_p - self.RHO_A) * self.GRAVITY * self.RHO_A / self.eta ** 2)
            * (D ** 2 / Ap)
            * (qe) ** (3 / 4)
        )
        # print(rho_p)
        # print('X, m, D, Ap', X, m, D, Ap)
        return X

    def best_number_Mitchell(self, D):
        """
        From Mitchell and Heymsfield 2005
        only using coeff from other studies,
        not taking into account IPAS values except Dmax
        """
        X = (
            2
            * self.alpha
            * self.GRAVITY
            * self.RHO_A
            * D ** (self.beta + 2 - self.sigma)
        ) / (self.gamma * self.eta ** 2)
        return X

    def reynolds_number(self, X):
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
            a = 0
            b = 1.0
        # return 8.5 * ((1 + 0.1519 * X ** (1 / 2)) ** (1 / 2) - 1) ** 2
        # print('a, x, b', X)
        return a * X ** b

    def dynamic_viscosity(self):
        # only true with T < 0C
        eta = 1.718 + 0.0049 * self.T - 1.2e-5 * self.T ** 2
        self.eta = eta * 1e-4 * (100 / 1000)

    def kinematic_viscosity(self):
        # return (0.000001458 * self.T ** (3 / 2)) / (self.T + 110.4)
        return self.eta / self.RHO_A
        # return 1.81E-5
        # should be around 1.17E-5 m2/s

    def terminal_velocity_Mitchell(self, D, Re):
        self.vt = (self.eta * Re) / (self.RHO_A * D)
