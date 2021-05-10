"""
Main function for running ice particle simulations
ICE-AGG collection
"""

import ipas.collection_no_db.crystal as crys
import ipas.collection_no_db.calculations as clus
import copy as cp
import numpy as np


def collect_clusters_iceagg(phio, r, ncrystals,
                            rand_orient, plot=False):

    # NEW AGGREGATE PROPERTIES
    agg_as = np.empty(ncrystals-1)
    agg_bs = np.empty(ncrystals-1)
    agg_cs = np.empty(ncrystals-1)
    cplxs = np.empty(ncrystals-1)
    phi2Ds = np.empty(ncrystals-1)
    phi2D = np.empty(ncrystals-1)
    dds = np.empty(ncrystals-1)

    # a and c axes of monomer using phi and r
    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    if c < a:
        plates = True
    else:
        plates = False

    # create Crystal
    crystal1 = crys.Crystal(a, c)
    crystal1.hold_clus = crystal1.points
    crystal1.orient_crystal(rand_orient)
    crystal1.recenter()

    # create cluster from initialized crystal
    # same orientation if crystal was reoriented
    cluster = clus.ClusterCalculations(crystal1)

    l=0
    # number of monomers/crystals per aggregate/cluster
    while cluster.ncrystals < ncrystals:
        crystal2 = crys.Crystal(a,c)  # initialize another monomer 
        crystal2.hold_clus = crystal2.points
        crystal2.orient_crystal(rand_orient)
        crystal2.recenter()

        # move monomer on top of cluster
        agg_pt, new_pt = cluster.generate_random_point_fast(crystal2, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        crystal2.move([-movediffx, -movediffy, 0])

        # move particles together
        cluster.combine(crystal2)

        # ----------- DENSITY CHANGE ----------
        # get cluster ellipsoid axes before aggregation
        A = cluster.fit_ellipsoid()
        cluster.ellipsoid_axes_lengths(A)
        # volume of ellipsoid around cluster before aggregation
        Ve_clus = 4./3.*np.pi*cluster.a*cluster.b*cluster.c 

        # a and c of monomers in cluster (all identical)
        a_clus = np.power((np.power(cluster.mono_r,3)/cluster.mono_phi),(1./3.))
        c_clus = cluster.mono_phi*a_clus
        # volume of all monomers in cluster
        Va_clus = 3*(np.sqrt(3)/2) * np.power(a_clus,2) * c_clus * cluster.ncrystals

        # density ratio of aggregate and ellipsoid
        d1 = Va_clus/Ve_clus
        # ------------------
        # add monomer points to original cluster (i.e., aggregate)
        cluster.add_crystal(crystal2)
        # save original points before reorienting for max area
        cluster.add_points = cp.deepcopy(cluster.points)
        # ------------------
        # monomer a and c axes
        a_mono = np.power((np.power(crystal2.r,3)/crystal2.phi),(1./3.))
        c_mono = crystal2.phi*a_mono
        # volume of monomer to collect
        Va_mono = 3*(np.sqrt(3)/2) * np.power(a_mono,2) * c_mono

        #get fit-ellipsoid radii (a-major, c-minor) after aggregation
        A = cluster.fit_ellipsoid()
        cluster.ellipsoid_axes_lengths(A)
        agg_as[l] = cluster.a
        agg_bs[l] = cluster.b
        agg_cs[l] = cluster.c

        # volume of ellipsoid around cluster after aggregation
        Ve_clus = 4./3.*np.pi*cluster.a*cluster.b*cluster.c
        d2 = (Va_clus + Va_mono)/Ve_clus
        # append relative change in density (after - before adding monomer)
        dds[l] = (d2-d1)/d1

        # ----------------------------
        # orient cluster after adding monomer
        if a > c and rand_orient == False:
            cluster.orient_cluster()
        else:
            cluster.orient_cluster(rand_orient)
        cluster.recenter()

        # ------other calculations------
        # save points before reorienting to calculate phi_2D_rotate
        cluster.orient_points = cp.deepcopy(cluster.points)
        cplxs[l], circle = cluster.complexity()
        phi2Ds[l] = cluster.phi_2D_rotate()
        phi2D[l] = cluster.phi_2D()
        # reset points back to how they were before phi_2D_rotate
        cluster.points = cluster.orient_points

        # -------- PLOTTING --------
        if plot:
            cluster.plot_ellipsoid_aggs([cluster, crystal2], view='z', circle=None, agg_agg=False)
            cluster.plot_ellipsoid_aggs([cluster, crystal2], view='x', circle=None, agg_agg=False)
            cluster.plot_ellipsoid_aggs([cluster, crystal2], view='y', circle=None, agg_agg=False)
            cluster.plot_ellipsoid_aggs([cluster, crystal2], view='w', circle=None, agg_agg=False)

        cluster_cp = cp.deepcopy(cluster)
        l+=1

    # characteristic values determined in postprocessing
    return agg_as, agg_bs, agg_cs, phi2D, cplxs, dds