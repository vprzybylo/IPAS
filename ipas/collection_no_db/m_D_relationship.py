"""
Main function for running ice particle simulations for m-D and vT calculations
ICE-AGG collection
"""

import copy as cp

import numpy as np

import ipas.collection_no_db.calculations as clus
import ipas.collection_no_db.crystal as crys


def collect_clusters_iceagg(phio, r, ncrystals, rand_orient, plot=False):

    # NEW AGGREGATE PROPERTIES
    agg_as = np.empty(ncrystals - 1)
    agg_bs = np.empty(ncrystals - 1)
    agg_cs = np.empty(ncrystals - 1)
    Aps = np.empty(ncrystals - 1)  # projected area of agg
    Acs = np.empty(ncrystals - 1)  # area of circumscribed circle
    Vps = np.empty(ncrystals - 1)  # volume of polygons of agg
    Ves = np.empty(ncrystals - 1)  # volume of circumscribed ellipsoid
    Dmaxs = np.empty(ncrystals - 1)  # maximum dimension of agg

    # a and c axes of monomer using phi and r
    a = (r ** 3 / phio) ** (1.0 / 3.0)
    c = phio * a

    # create Crystal
    crystal1 = crys.Crystal(a, c)
    crystal1.hold_clus = crystal1.points
    crystal1.orient_crystal(rand_orient)
    crystal1.recenter()

    # create cluster from initialized crystal
    # same orientation if crystal was reoriented
    cluster = clus.ClusterCalculations(crystal1)

    l = 0
    # number of monomers/crystals per aggregate/cluster
    while cluster.ncrystals < ncrystals:
        crystal2 = crys.Crystal(a, c)  # initialize another monomer
        crystal2.hold_clus = crystal2.points
        crystal2.orient_crystal(rand_orient)
        crystal2.recenter()

        # move monomer on top of cluster
        agg_pt, new_pt = cluster.generate_random_point_fast(crystal2, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        crystal2.move([-movediffx, -movediffy, 0])

        # move particles together
        # cluster.combine(crystal2)
        cluster.add_crystal_from_above(crystal2)

        # add monomer points to original cluster (i.e., aggregate)
        cluster.add_crystal(crystal2)
        # save original points before reorienting for max area
        cluster.add_points = cp.deepcopy(cluster.points)
        # ------------------

        # a and c of monomers in cluster (all identical)
        a_clus = np.power((np.power(cluster.mono_r, 3) / cluster.mono_phi), (1.0 / 3.0))
        c_clus = cluster.mono_phi * a_clus
        # volume of all monomers in cluster
        V_clus = 3 * (np.sqrt(3) / 2) * np.power(a_clus, 2) * c_clus * cluster.ncrystals
        Vps[l] = V_clus

        # get fit-ellipsoid radii (a-major, c-minor) after aggregation
        A = cluster.fit_ellipsoid()
        cluster.ellipsoid_axes_lengths(A)
        agg_as[l] = cluster.a
        agg_bs[l] = cluster.b
        agg_cs[l] = cluster.c

        # volume of ellipsoid around cluster after aggregation
        Ve_clus = 4.0 / 3.0 * np.pi * cluster.a * cluster.b * cluster.c
        Ves[l] = Ve_clus

        # ----------------------------
        # orient cluster after adding monomer
        if a > c and rand_orient is False:
            cluster.orient_cluster()
        else:
            cluster.orient_cluster(rand_orient)
        cluster.recenter()
        cluster.orient_points = cp.deepcopy(cluster.points)

        # ----------------------------
        _, _ = cluster.complexity()
        Aps[l] = cluster.Ap
        Acs[l] = cluster.Ac  # area of encompassing circle
        cluster.farthest_points()
        Dmaxs[l] = cluster.max_dimension()  # 3D

        if plot:
            cluster.plot_ellipsoid_aggs(
                [cluster, crystal2], view="z", circle=None, agg_agg=False
            )
            cluster.plot_ellipsoid_aggs(
                [cluster, crystal2], view="x", circle=None, agg_agg=False
            )
            cluster.plot_ellipsoid_aggs(
                [cluster, crystal2], view="y", circle=None, agg_agg=False
            )
            cluster.plot_ellipsoid_aggs(
                [cluster, crystal2], view="w", circle=None, agg_agg=False
            )

        l += 1

    # characteristic values determined in postprocessing
    return agg_as, agg_bs, agg_cs, Aps, Acs, Vps, Ves, Dmaxs
