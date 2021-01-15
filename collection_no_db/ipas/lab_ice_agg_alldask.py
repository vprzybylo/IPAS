"""Utilities for running ice particle simulations."""
import copy as cp
import numpy as np
import ipas 
#from .db import Session
import time
import logging
import multiprocessing
import pandas as pd
import pickle

       
def collect_clusters(phio, r, ncrystals, rand_orient):
    
    cplxs = np.empty(ncrystals-1)
    agg_as = np.empty(ncrystals-1)
    agg_bs = np.empty(ncrystals-1)
    agg_cs = np.empty(ncrystals-1)
    phi2Ds = np.empty(ncrystals-1) 
    phi2D = np.empty(ncrystals-1) 
    dds = np.empty(ncrystals-1)
    perims = np.empty(ncrystals-1)
    
    #a = (r ** 3 / phio) ** (1. / 3.)
    #c = phio * a
    
    if phio < 1.0:
        a=r
        c=phio*a

    else:
        c=r
        a=c/phio
    
    crystal1 = ipas.Ice_Crystal(a, c)
    crystal1.hold_clus = crystal1.points
    crystal1.orient_crystal(rand_orient)
    crystal1.recenter()
    
    
    #cluster will start with a random orientation if crystal was reoriented
    cluster = ipas.Cluster_Calculations(crystal1)          
    l=0

    while cluster.ncrystals < ncrystals: 
        crystal2 = ipas.Ice_Crystal(a,c)
        crystal2.hold_clus = crystal2.points
        crystal2.orient_crystal(rand_orient)
        crystal2.recenter()

        #start collection
        agg_pt, new_pt = cluster.generate_random_point_fast(crystal2, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        crystal2.move([-movediffx, -movediffy, 0])

        cluster.closest_points(crystal2)

        #for density change of cluster before aggregating
        rx,ry,rz = cluster.spheroid_axes()  
        Ve_clus = 4./3.*np.pi*rx*ry*rz 
        a_clus=np.power((np.power(cluster.mono_r,3)/cluster.mono_phi),(1./3.))
        c_clus = cluster.mono_phi*a_clus
       
        Va_clus = 3*(np.sqrt(3)/2) * np.power(a_clus,2) * c_clus * cluster.ncrystals

        
        d1 = Va_clus/Ve_clus

        cluster.add_crystal(crystal2)
        cluster.add_points = cp.deepcopy(cluster.points)

        if a>c and rand_orient== False:
            cluster.orient_cluster() 
        else:
            cluster.orient_cluster(rand_orient) 
        cluster.recenter()
        cluster.orient_points = cp.deepcopy(cluster.points)

#         depths[l] = cluster.depth()
#         major_ax_zs[l] = cluster.major_ax('z')
        agg_a, agg_b, agg_c = cluster.spheroid_axes()  
        agg_as[l] = agg_a
        agg_bs[l] = agg_b
        agg_cs[l] = agg_c

        #DENSITY CHANGE formed agg ------------------
        a_mono = np.power((np.power(crystal2.r,3)/crystal2.phi),(1./3.))
        c_mono = crystal2.phi*a_mono
        Va_mono = 3*(np.sqrt(3)/2) * np.power(a_mono,2) * c_mono
        Ve3 = 4./3.*np.pi*agg_a*agg_b*agg_c  #volume of ellipsoid for new agg
        d2 = (Va_clus+Va_mono)/Ve3
  
        dds[l] = (d2-d1)/d1
        #-------------------------------------

        #cplxs[l], perims[l] = cluster.complexity()

        #phi2Ds[l] = cluster.phi_2D_rotate()
        #phi2D[l] = cluster.phi_2D()
        cluster.points = cluster.orient_points

        #print(agg_c/agg_a)
#             print('w')
#        cluster.plot_ellipsoid_aggs([cluster, crystal2], view='w', circle=None, agg_agg=False)
#             print('x')
        
#         cluster.plot_ellipsoid_aggs([cluster, crystal2], view='x', circle=None, agg_agg=False)
#         print('y')
#         cluster.plot_ellipsoid_aggs([cluster, crystal2], view='w', circle=None, agg_agg=False)
#         print('z')
#         cluster.plot_ellipsoid_aggs([cluster, crystal2], view='z', circle=None, agg_agg=False)

        cluster_cp = cp.deepcopy(cluster)
        l+=1

    #print('made it to the end of collect_clusters loops')
    return agg_as, agg_bs, agg_cs, phi2D, dds