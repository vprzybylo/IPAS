"""
Runs ice-aggregate collection
"""

import ipas.collection_from_db.crystal as crys
import time
import numpy as np
import random
import copy as cp


def collect_clusters_iceagg(a, c, clusters, rand_orient=False, plot=False):

    #NEW AGGREGATE PROPERTIES
    cplxs = []
    agg_as = []
    agg_bs = []
    agg_cs = []
    phi2Ds = []
    dds = []

    # how many aggregates to create
    for n in range(len(clusters)):
        # create Crystal
        monomer = crys.Crystal(a[n], c[n])
        monomer.hold_clus = monomer.points
        monomer.orient_crystal(rand_orient)
        monomer.recenter()

        # get cluster (agg) from db
        cluster = clusters[n]

        # orient original cluster and recenter
        if a[n]>c[n] and rand_orient== False:
            cluster.orient_cluster()
        else:
            cluster.orient_cluster(rand_orient) 
        cluster.recenter()

        # move monomer on top of cluster
        agg_pt, new_pt = cluster.generate_random_point_fast(monomer, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        monomer.move([-movediffx, -movediffy, 0])

        # move cluster and monomer together
        cluster.closest_points(monomer)

        # ----------- DENSITY CHANGE ----------
        # get cluster ellipsoid axes before aggregation
        A = cluster.fit_ellipsoid()
        cluster.ellipsoid_axes_lengths(A)
        # volume of ellipsoid around cluster before aggregation
        Ve_clus = 4./3.*np.pi*cluster.a*cluster.b*cluster.c 

        # a and c of monomers in cluster (all identical)
        a_clus = np.power((np.power(cluster.monor,3)/cluster.monophi),(1./3.))
        c_clus = cluster.monophi*a_clus
        # volume of all monomers in cluster
        Va_clus = 3*(np.sqrt(3)/2) * np.power(a_clus,2) * c_clus * cluster.ncrystals

        # density ratio of aggregate and ellipsoid
        d1 = Va_clus/Ve_clus

        # -------------------
        # add monomer points to original cluster (i.e., aggregate)
        cluster.add_crystal(monomer)
        # save original points before reorienting for max area
        cluster.add_points = cp.deepcopy(cluster.points)
        # -------------------

        # monomer a and c axes
        a_mono = np.power((np.power(monomer.r,3)/monomer.phi),(1./3.))
        c_mono = monomer.phi*a_mono
        # volume of monomer to collect
        Va_mono = 3*(np.sqrt(3)/2) * np.power(a_mono,2) * c_mono

        # get fit-ellipsoid radii (a-major, c-minor) after aggregation
        A = cluster.fit_ellipsoid()
        cluster.ellipsoid_axes_lengths(A)
        agg_as.append(cluster.a)
        agg_bs.append(cluster.b)
        agg_cs.append(cluster.c)

        # volume of ellipsoid around cluster after aggregation
        Ve_clus = 4./3.*np.pi*cluster.a*cluster.b*cluster.c
        d2 = (Va_clus + Va_mono)/Ve_clus
        # append relative change in density (after - before adding monomer)
        dds.append((d2 - d1)/d1)

        # ----------------------------
        # orient cluster after adding monomer
        if a[n] > c[n] and rand_orient == False:
            cluster.orient_cluster()
        else:
            cluster.orient_cluster(rand_orient)
        cluster.recenter()

        # ------other calculations------
        # save points before reorienting to calculate phi_2D_rotate
        cluster.orient_points = cp.deepcopy(cluster.points)
        cplx, circle= cluster.complexity()
        cplxs.append(cplx)
        phi2Ds.append(cluster.phi_2D_rotate())
        # reset points back to how they were before phi_2D_rotate
        cluster.points = cluster.orient_points

        # -------- PLOTTING --------
        if plot:
            cluster.plot_ellipsoid_aggs([cluster, monomer], view='x', circle=None, agg_agg=False)
            cluster.plot_ellipsoid_aggs([cluster], view='z', circle=None, agg_agg=False)
           cluster.plot_ellipsoid_aggs([cluster, monomer], view='y', circle=None, agg_agg=False)
           cluster.plot_ellipsoid_aggs([cluster, monomer], view='z', circle=None, agg_agg=False)

    # characteristic values determined in postprocessing
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds
