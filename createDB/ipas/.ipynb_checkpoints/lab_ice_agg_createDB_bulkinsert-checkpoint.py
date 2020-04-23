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
    start = time.time()
    
    list_of_clusters = []
    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    if c < a:
        plates = True
    else:
        plates = False

    count = 0
    for n in range(nclusters):
        if n % 2 == 0.:
            print('nclus',int(np.round(r)), phio, n)

        crystal1 = ipas.Ice_Crystal(c, a)
        crystal1.hold_clus = crystal1.points
        crystal1.orient_crystal(rand_orient)
        crystal1.recenter()
        cluster = ipas.Cluster_Calculations(crystal1)  #cluster will start with a random orientation if crystal was reoriented

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

            cluster.add_crystal(crystal2)
            cluster.add_points = cp.deepcopy(cluster.points)

            if a>c and rand_orient== False:
                cluster.orient_cluster() 
            else:
                cluster.orient_cluster(rand_orient) 
            cluster.recenter()
            cluster.orient_points = cp.deepcopy(cluster.points)
            
            cluster.spheroid_axes()  # radii lengths - 3 axes
            cluster.complexity()
            cluster.phi_2D_rotate()
             
#             print('w')
#             cluster.plot_ellipsoid_aggs([cluster], view='w', circle=None)
#             print('x')
#             cluster.plot_ellipsoid_aggs([cluster], view='x', circle=None)
#             print('y')
#             cluster.plot_ellipsoid_aggs([cluster], view='y', circle=None)
#             print('z')
#             cluster.plot_ellipsoid_aggs([cluster], view='z', circle=None)

            cluster_cp = cp.deepcopy(cluster)
#             cluster_cp.xs = [p["x"] for p in cluster_cp.points.ravel()]
#             cluster_cp.ys = [p["y"] for p in cluster_cp.points.ravel()]
#             cluster_cp.zs = [p["z"] for p in cluster_cp.points.ravel()]
            #cluster_cp.points = pickle.dump(cluster_cp.points)
            #print(cluster_cp.points)
            list_of_clusters.append(cluster_cp)   
   
    end = time.time()            
    print('made it to the end of collect_clusters loops', (end-start))
        
    ds = [clus.to_dict() for clus in list_of_clusters]
    d = {}
    for k in ds[0].keys():
        d[k] = tuple(d[k] for d in ds)
    return d