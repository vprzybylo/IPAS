"""main function for running ice particle simulations."""

import copy as cp
import numpy as np
import ipas 
import time
import logging
import multiprocessing
import pandas as pd
import pickle


def collect_clusters_alldask(phio, r, ncrystals, rand_orient):

    cplxs = np.empty(ncrystals-1)
    agg_as = np.empty(ncrystals-1)
    agg_bs = np.empty(ncrystals-1)
    agg_cs = np.empty(ncrystals-1)
    phi2Ds = np.empty(ncrystals-1)
    phi2D = np.empty(ncrystals-1)
    dds = np.empty(ncrystals-1)

    # a and c axes of monomer
    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    if c < a:
        plates = True
    else:
        plates = False

    crystal1 = ipas.IceCrystal(a, c)  # initialize monomer 1
    crystal1.hold_clus = crystal1.points
    crystal1.orient_crystal(rand_orient)
    crystal1.recenter()

    # cluster will start with a random orientation if crystal was reoriented
    cluster = ipas.ClusterCalculations(crystal1)
    l=0

    while cluster.ncrystals < ncrystals:
        crystal2 = ipas.IceCrystal(a,c)  # initialize another monomer 
        crystal2.hold_clus = crystal2.points
        crystal2.orient_crystal(rand_orient)
        crystal2.recenter()

        # move monomer on top of cluster
        agg_pt, new_pt = cluster.generate_random_point_fast(crystal2, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        crystal2.move([-movediffx, -movediffy, 0])

        # move particles together
        cluster.closest_points_old(crystal2)

        # ----------- DENSITY CHANGE ----------
        # get cluster ellipsoid axes before aggregation
        rx,ry,rz = cluster.ellipsoid_axes()
        # volume of ellipsoid around cluster before aggregation
        Ve_clus = 4./3.*np.pi*rx*ry*rz 

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
        agg_a, agg_b, agg_c = cluster.ellipsoid_axes()
        agg_as[l] = agg_a
        agg_bs[l] = agg_b
        agg_cs[l] = agg_c
        #print(a, agg_cs)
        
        # volume of ellipsoid around cluster after aggregation        
        Ve_clus = 4./3.*np.pi*agg_a*agg_b*agg_c
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
#         print('AFTER')
#         cluster.plot_ellipsoid_aggs([cluster, crystal2], view='z', circle=None, agg_agg=False)
#         cluster.plot_ellipsoid_aggs([cluster, crystal2], view='x', circle=None, agg_agg=False)
#         cluster.plot_ellipsoid_aggs([cluster, crystal2], view='y', circle=None, agg_agg=False)
        cluster.plot_ellipsoid_aggs([cluster, crystal2], view='w', circle=None, agg_agg=False)

        cluster_cp = cp.deepcopy(cluster)
        l+=1

    #print('made it to the end of collect_clusters loops')
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds