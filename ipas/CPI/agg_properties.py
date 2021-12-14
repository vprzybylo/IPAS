"""
calculates geometric parameters for IPAS aggregates to be used in CPI verification figure
"""

import sys

sys.path.append("../collection_from_db")

import numpy as np
import shapely.geometry as geom
import shapely.ops as shops
from shapely.geometry import Point

import ipas.cluster_calculations as cc


class Agg:
    def __init__(self, cluster, dims=["x", "z"]):
        self.ncrystals = cluster.ncrystals
        self.points = cluster.points
        self.a = cluster.a
        self.b = cluster.b
        self.c = cluster.c
        self.monor = cluster.mono_r
        self.monophi = cluster.mono_phi
        self.r = cluster.agg_r
        self.phi = cluster.agg_phi
        polygons = [
            geom.MultiPoint(self.points[n][dims]).convex_hull
            for n in range(self.ncrystals)
        ]
        self.agg = shops.cascaded_union(polygons)
        self.area = self.agg.area
        self.perim = self.agg.length

        # call all functions below
        self.filled_circular_area_ratio(cluster)
        self.convex_perim()  # before roundness
        self.circularity()
        self.roundness()
        self.perim_area_ratio()
        self.convexity()
        self.complexity(cluster)
        self.hull_area()
        self.solidity()
        self.equiv_d()

    def get_list(self):
        return {
            "ncrystals": self.ncrystals,
            "monor": self.monor,
            "monophi": self.monophi,
            "r": self.r,
            "phi": self.phi,
            "area_ratio": self.filled_circ_area_ratio,
            "convex_perim": self.convex_perim,
            "circularity": self.circularity,
            "roundness": self.roundness,
            "perim_area_ratio": self.perim_area_ratio,
            "convexity": self.convexity,
            "complexity": self.complexity,
            "hull_area": self.hull_area,
            "solidity": self.solidity,
            "equiv_d": self.equiv_d,
        }

    def filled_circular_area_ratio(self, cluster):
        """returns the area of the largest contour divided by the area of
        an encompassing circle

        useful for spheres that have reflection spots that are not captured
        by the largest contour and leave a horseshoe pattern"""

        poly = shops.cascaded_union(self.agg).convex_hull
        x, y = poly.exterior.xy
        c = cc.Cluster_Calculations(cluster)
        circ = c.make_circle([x[i], y[i]] for i in range(len(x)))
        circle = Point(circ[0], circ[1]).buffer(circ[2])
        x, y = circle.exterior.xy
        Ac = circle.area
        self.filled_circ_area_ratio = self.area / Ac

    def circularity(self):
        self.circularity = (4.0 * np.pi * self.area) / (self.perim ** 2)

    def roundness(self):
        """similar to circularity but divided by the perimeter
        that surrounds the largest contour squared instead of the
        actual convoluted perimeter"""

        self.roundness = (4.0 * np.pi * self.area) / self.convex_perim ** 2

    def perim_area_ratio(self):
        self.perim_area_ratio = self.perim / self.area

    def convex_perim(self):
        """returns the perimeter of the convex hull of the
        largest contour
        """
        self.convex_perim = self.agg.convex_hull.length

    def convexity(self):
        self.convexity = self.convex_perim / self.perim

    def complexity(self, cluster):
        """similar to the fractal dimension of the particle

        see:
            Schmitt, C. G., and A. J. Heymsfield (2014),
            Observational quantification of the separation of
            simple and complex atmospheric ice particles,
            Geophys. Res. Lett., 41, 1301–1307, doi:10.1002/ 2013GL058781.
        """
        c = cc.Cluster_Calculations(cluster)
        self.complexity, _ = c.complexity()
        # self.complexity = 10*(0.1-(self.area/(np.sqrt(self.area/self.hull_area)*self.perim**2)))

    def hull_area(self):
        """area of a convex hull surrounding the largest contour"""
        self.hull_area = self.agg.convex_hull.area

    def solidity(self):
        self.solidity = self.area / self.hull_area

    def equiv_d(self):
        """equivalent diameter of a circle with the same area as the largest contour"""
        self.equiv_d = np.sqrt(4 * self.area / np.pi)
