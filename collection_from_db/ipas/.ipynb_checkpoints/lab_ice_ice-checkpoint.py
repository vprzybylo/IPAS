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

       
def collect_clusters(phio, r, nclusters, ncrystals, rand_orient):
    
    #NEW AGGREGATE PROPERTIES
    cplxs = []
    agg_as = []
    agg_bs = []
    agg_cs = []
    phi2Ds = []    
    dds = []
    depths = []
    major_ax_zs = []
    
    list_of_clusters = []
    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    if c < a:
        plates = True
    else:
        plates = False

    count = 0
    for n in range(nclusters):
        if n % 20 == 0.:
            print('nclus',int(np.round(r)), phio, n)

        crystal1 = ipas.Ice_Crystal(c, a)
        crystal1.hold_clus = crystal1.points
        crystal1.orient_crystal(rand_orient)
        crystal1.recenter()
        #cluster will start with a random orientation if crystal was reoriented
        cluster = ipas.Cluster_Calculations(crystal1)  
        
        while cluster.ncrystals < ncrystals: 
            
            crystal2 = ipas.Ice_Crystal(c,a)
            crystal2.hold_clus = crystal2.points
            crystal2.orient_crystal(rand_orient)
            crystal2.recenter()

            #start collection
            agg_pt, new_pt = cluster.generate_random_point_fast(crystal2, 1)
            movediffx = new_pt.x - agg_pt.x
            movediffy = new_pt.y - agg_pt.y
            crystal2.move([-movediffx, -movediffy, 0])

            cluster.closest_points(crystal2)
            
            #starting volume ratio between monomer and fit ellipsoid
            a1=np.power((np.power(crystal1.r,3)/crystal1.phi),(1./3.))
            c1= crystal1.phi*a1
            Va = 3*(np.sqrt(3)/2) * np.power(a1,2) * c1 
            a_crys_ell,b_crys_ell,c_crys_ell= crystal1.spheroid_axes()
            Ve = 4./3.*np.pi*a_crys_ell*b_crys_ell*c_crys_ell 
            d1 = Va/Ve 

            cluster.add_crystal(crystal2)
            cluster.add_points = cp.deepcopy(cluster.points)

            if a>c and rand_orient== False:
                cluster.orient_cluster() 
            else:
                cluster.orient_cluster(rand_orient) 
            cluster.recenter()
            cluster.orient_points = cp.deepcopy(cluster.points)
            
            cluster.spheroid_axes()  # radii lengths - 3 axes
            
            depths.append(cluster.depth())
            major_ax_zs.append(cluster.major_ax('z'))
            agg_a, agg_b, agg_c = cluster.spheroid_axes()  
            agg_as.append(agg_a)
            agg_bs.append(agg_b)
            agg_cs.append(agg_c)
            
            #FOR DENSITY CHANGE ------------------
            #monomer a and c
            a1=np.power((np.power(crystal1.r,3)/crystal1.phi),(1./3.))
            c1= crystal1.phi*a1
            Va = 3*(np.sqrt(3)/2) * np.power(a1,2) * c1 *cluster.ncrystals
            Ve = 4./3.*np.pi*cluster.agg_a*cluster.agg_b*cluster.agg_c 
            d2 = Va/Ve 
            dd=d2-d1
            dds.append(dd) #change in density
            #print(d1, d2, dd)
            d1=d2 
      
            #-------------------------------------

            cluster.complexity()
            cplx, circle= cluster.complexity()
            
            cplxs.append(cplx)
            phi2Ds.append(cluster.phi_2D())
            cluster.points = cluster.orient_points
            
#             print('w')
#             cluster.plot_ellipsoid_aggs([cluster], view='w', circle=None)
#             print('x')
#             cluster.plot_ellipsoid_aggs([cluster], view='x', circle=None)
#             print('y')
#             cluster.plot_ellipsoid_aggs([cluster], view='y', circle=None)
#             print('z')
#            cluster.plot_ellipsoid_aggs([cluster], view='z', circle=None)

            cluster_cp = cp.deepcopy(cluster)
            list_of_clusters.append(cluster_cp)   


    print('made it to the end of collect_clusters loops')
    return agg_as, agg_bs, agg_cs, major_ax_zs, depths, phi2Ds, cplxs, dds