"""Runs ice-aggregate collection."""

import ipas
import time
import numpy as np
import random
import copy as cp


def collect_clusters_ice_agg(a, c, clusters, rand_orient=False):


    #NEW AGGREGATE PROPERTIES
    cplxs = []
    agg_as = []
    agg_bs = []
    agg_cs = []
    phi2Ds = []
    dds = []

    for n in range(len(clusters)):
        monomer = ipas.IceCrystal(a[n], c[n])
        monomer.hold_clus = monomer.points
        monomer.orient_crystal(rand_orient)
        monomer.recenter() 
        cluster = clusters[n]

        if a[n]>c[n] and rand_orient== False:
            cluster.orient_cluster()
        else:
            cluster.orient_cluster(rand_orient) 
        cluster.recenter()
        #cluster.orient_points = cp.deepcopy(cluster.points)

        agg_pt, new_pt = cluster.generate_random_point_fast(monomer, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        monomer.move([-movediffx, -movediffy, 0])

        cluster.closest_points(monomer)

        A = cluster.fit_ellipsoid()
        cluster.ellipsoid_axes_lengths(A)    
        agg_as.append(cluster.a)
        agg_bs.append(cluster.b)
        agg_cs.append(cluster.c)
        
        #for density change
        Ve_clus = 4./3.*np.pi*cluster.a*cluster.b*cluster.c  #volume of ellipsoid for new agg
        a_clus=np.power((np.power(cluster.monor,3)/cluster.monophi),(1./3.))
        c_clus= cluster.monophi*a_clus
        Va_clus = 3*(np.sqrt(3)/2) * np.power(a_clus,2) * c_clus * cluster.ncrystals

        a_mono=np.power((np.power(monomer.r,3)/monomer.phi),(1./3.))
        c_mono= monomer.phi*a_mono
        Va_mono = 3*(np.sqrt(3)/2) * np.power(a_mono,2) * c_mono 

        a_mono_ell,b_mono_ell,c_mono_ell= monomer.spheroid_axes()
        d1 = Va_clus/Ve_clus

        cluster.add_crystal(monomer)
        cluster.add_points = cp.deepcopy(cluster.points)

        if a[n]>c[n] and rand_orient== False:
            cluster.orient_cluster()
        else:
            cluster.orient_cluster(rand_orient)
        cluster.recenter()
        cluster.orient_points = cp.deepcopy(cluster.points)

        #DENSITY CHANGE ------------------
        d2 = (Va_clus+Va_mono)/Ve_clus

        #print((d2-d1)/d1)
        dds.append((d2-d1)/d1)

        #-------------------------------------

        cplx, circle= cluster.complexity()
        cplxs.append(cplx)
        phi2Ds.append(cluster.phi_2D_rotate())
        cluster.points = cluster.orient_points

        #PLOTTING
        #cluster.plot_ellipsoid_aggs([cluster, monomer], view='x', circle=None, agg_agg=False)
        #cluster.plot_ellipsoid_aggs([cluster], view='z', circle=None, agg_agg=False)

#        cluster.plot_ellipsoid_aggs([cluster, monomer], view='y', circle=None, agg_agg=False)
#        cluster.plot_ellipsoid_aggs([cluster, monomer], view='z', circle=None, agg_agg=False)
    #characteristic values determined in postprocessing
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds
