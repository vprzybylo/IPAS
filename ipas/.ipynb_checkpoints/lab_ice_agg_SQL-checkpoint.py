"""Utilities for running ice particle simulations."""

import random
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as stats
#from scipy.stats import rv_continuous, gamma
import matplotlib
import shapely.geometry as geom
import scipy
import warnings
import pandas as pd
import statsmodels as sm
import math
from descartes import PolygonPatch
from shapely.geometry import Point
import cProfile
import dask
import copy as cp

#from ipas import IceClusterBatchIceAgg_Ntot as batch
from ipas import IceClusterAggAgg_SQL as clus
from ipas import IceCrystal as crys

from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy import create_engine, Table, MetaData
import base

def collect_clusters(session, engine, phio, r, nclusters, ncrystals, save_plots=False, numaspectratios=50, 
                     rand_orient=False, lodge=0, max_misses=20, ch_dist='gamma'):

    cplxs = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    rxs = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    rys = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    rzs = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    phi2Ds = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    list_of_clusters = []
    list_of_crystals = []
    
    connection = engine.connect()
    print('done connecting to database')

    width = (r**3/phio)**(1./3.)
    length=phio*width
    if length>width:
        plates = False
    else:
        plates = True
        
    #all monomers the same for now
    r = np.power((np.power(width,2)*length),(1./3.)) #of monomer
    
    f5 = open('outfile_allflat_rall.dat',"w")       

    k=0  #for saving plot constraints
    count = 0
    for n in range(nclusters):
        
        if n % 50 == 0:
            #print('nclus',r, phio, n)
            f5.write('%d\t %.3f\t %d\t\n'%(r, phio, n))
            f5.flush()
        
        crystal = crys.IceCrystal(length=length, width=width, rand_orient=rand_orient)
        crystal.orient_crystal(rand_orient)
        
        cluster1 = clus.IceCluster(crystal, n)
        cluster1.ncrystals = 1
        
        l=0  #crystals in cluster1
        while cluster1.ncrystals < ncrystals:  #agg 1

            new_crystal = crys.IceCrystal(length=length, width=width,rand_orient=rand_orient)
            new_crystal.orient_crystal(rand_orient)

            ck_intersect = False
            while ck_intersect is False:
                agg_pt, new_pt = cluster1.generate_random_point(new_crystal, 1)
                movediffx = new_pt.x - agg_pt.x
                movediffy = new_pt.y - agg_pt.y
                new_crystal.move([-movediffx, -movediffy, 0])

                '''NEW NEAREST PT METHOD for ice-agg'''
                #cluster2.closest_points(new_crystal)
                #ck_intersect = True
                #print('clus 2 add xtal from above',file=file) 
                ck_intersect = cluster1.add_crystal_from_above(new_crystal, lodge=lodge)               

            #cluster1.plot_ellipsoid()
            #print(cluster1.id)
            cluster1.add_crystal(new_crystal) 
            
            if rand_orient:
                cluster1.orient_cluster()#rand_orient =False
                    
            #cluster1.plot_ellipsoid()
            Va = 3*(np.sqrt(3)/2) * np.power(width,2) * length * cluster1.ncrystals  #actual agg volume of hexagonal prisms
            a,b,c = cluster1.spheroid_axes(plates)  #radii lengths - 3 axes

            cplxs[n,l]=cluster1.complexity
            rxs[n,l]=a
            rys[n,l]=b
            rzs[n,l]=c
            phi2Ds[n,l] = cluster1.phi_2D_rotate()
            
            cluster1cp = cp.deepcopy(cluster1)
            #list_of_crystals.append(new_crystal)
            list_of_clusters.append(cluster1cp)
            
#             try:
#                 session.add(new_crystal)
#                 session.add(cluster1cp)
#                 session.commit()
#             except:
#                 print('except')
#                 raise
            
                
            count +=1
            l+=1
            
    print('-'*30)
    for obj in list_of_clusters:
        print(obj.ncrystals, obj.agg_phi, obj.agg_r)
    print('len clus list, len crys list', len(list_of_clusters), len(list_of_crystals))
        
    #ADD TO DATABASE
    try:
        #session.add(new_crystal)
        session.add_all(list_of_clusters)
        session.add(new_crystal)
        session.commit()

    except:
         print('in except')
     #    session.rollback()
         raise

        
    print('made it to the end of collect_clusters loops')
    f5.close()
    

    return [phio, r, width, length, rxs, rys, rzs, phi2Ds, cplxs]
  