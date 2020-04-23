"""Utilities for running ice particle simulations."""

import random
import time
import numpy as np
import scipy.stats as stats
#from scipy.stats import rv_continuous, gamma
import matplotlib
import shapely.geometry as geom
import scipy
import pandas as pd
import statsmodels as sm
import math
from shapely.geometry import Point
import cProfile

from ipas import plots_phiarr as plts
from ipas import IceClusterBatchAggAgg_Ntot as batch
from ipas import IceCrystal as crys
from ipas import IceClusterAggAgg as clus


def collect_clusters(phio, r, nclusters, ncrystals1, ncrystals2, save_plots=False, numaspectratios=50, 
                     rand_orient=False, lodge=0, max_misses=20, ch_dist='gamma'):

    """Simulate crystal aggregates.

    Args:
        length (float): The column length of the crystals.
        width (float): The width of the hexagonal faces of the crystals.
        nclusters (int): The number of clusters to simulate.
        ncrystals (int): The number of crystals in each cluster.
        rand_orient (bool): If true, randomly orient the crystals and aggregate.
            Uses the first random orientation and sets speedy to False.
            Default is True.
        numaspectratios (int): The number of monomer aspect ratios to loop over.
            Default is 50.
        reorient (str): The method to use for reorienting crystals and clusters.
            'random' chooses rotations at random and selects the area-maximizing
            rotation. 'IDL' exactly reproduces the IPAS IDL code. Default is
            'random'. <-- should leave as is ('IDL' was for testing purposes).x
        rotations (int): The number of rotations to use to reorient crystals and
            clusters. Default is 50.
        speedy (bool): If true, choose an optimal rotation for single crystals
            instead of reorienting them. Default is true.
        lodge (float): The vertical distance that crystals lodge into each other
            when added from above. Useful for matching the output of IPAS IDL code,
            which uses 0.5. Default is zero.

    Returns:
        An IceClusterBatch object containing the simulated clusters.
       
    """

    
    #AGGREGATE PROPERTIES
    cplxs = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    rxs = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    rys = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    rzs = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    phi2Ds = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    phioarr = []
    rarr = []
    lengtharr = []
    widtharr = []
    
    #phioarr.append(phio)
    width = (r**3/phio)**(1./3.)
    #widtharr.append(width)
    length=phio*width
    #lengtharr.append(length)
    #all monomers the same for now
    r = np.power((np.power(width,2)*length),(1./3.)) #of monomer
    #rarr.append(r)
    
    k=0  #for saving plot constraints
    
    for n in range(nclusters):

        if n % 50 == 0:
            print('nclus',r, phio, n)
            f5.write('%d\t %.3f\t %d\t\n'%(r, phio, n))
            f5.flush()
        #print('nclus',n)
        plates = width > length
       
        # initialize cluster 1 and 2
        crystal = crys.IceCrystal(length=length, width=width)
        crystal.orient_crystal(rand_orient)
        cluster1 = clus.IceCluster(crystal)
        
        count1 = 0
        count2 = 0
        l=0  #crystals in cluster1
        fail=0
        while cluster1.ncrystals < ncrystals1:  #agg 1

            #start_ice = time.time()
            #print('n,l+m',cluster1.ncrystals,cluster2.ncrystals, ncrystals1)

            if count1 > 0:
                
                if cluster1.ncrystals == 1:
                    l=1                

                new_crystal = crys.IceCrystal(length=length, width=width)
                new_crystal.orient_crystal(rand_orient)

                ck_intersect = False
                while ck_intersect is False:
                #print('clus 2 rand pt',file=file) 
                    agg_pt, new_pt = cluster1.generate_random_point(new_crystal, 1)
                    movediffx = new_pt.x - agg_pt.x
                    movediffy = new_pt.y - agg_pt.y
                    new_crystal.move([-movediffx, -movediffy, 0])

                    '''NEW NEAREST PT METHOD for ice-agg'''
                    #cluster2.closest_points(new_crystal)
                    #ck_intersect = True
                    #print('clus 2 add xtal from above',file=file) 
                    ck_intersect = cluster1.add_crystal_from_above(new_crystal, lodge=lodge) 

                cluster1.add_crystal(new_crystal)
                       
            crystal = crys.IceCrystal(length=length, width=width)
            crystal.orient_crystal(rand_orient)
            cluster2 = clus.IceCluster(crystal)

            #cluster2.plot_ellipsoid_agg_agg(cluster2)
            m=0  #crystals in cluster2
            while cluster2.ncrystals < ncrystals2:  #agg 2
              
                if cluster2.ncrystals == 1:
                    m=1
                
                if count2 > 0:
                                # make a new crystal, orient it
                    new_crystal = crys.IceCrystal(length=length, width=width)
                    new_crystal.orient_crystal(rand_orient)
                    if rand_orient:
                        rand_orient = False
                        cluster2.orient_cluster(rand_orient)
                        rand_orient = True
                        
                    ck_intersect = False
                    while ck_intersect is False:
                        agg_pt, new_pt = cluster2.generate_random_point(new_crystal, 1)
                        movediffx = new_pt.x - agg_pt.x
                        movediffy = new_pt.y - agg_pt.y
                        new_crystal.move([-movediffx, -movediffy, 0])

                        '''NEW NEAREST PT METHOD for ice-agg'''
                        #cluster2.closest_points(new_crystal)
                        #ck_intersect = True
                        #print('clus 2 add xtal from above',file=file) 
                        ck_intersect = cluster2.add_crystal_from_above(new_crystal, lodge=lodge)               
                    
                    cluster2.add_crystal(new_crystal)
                    #cluster2.plot_ellipsoid()

                else:
                    m=0  #to start index for dd at 0 for 1 crystal (ice-ice)

                '''----AGG-AGG COLLECTION ------'''
                
                #start_agg = time.time()
                #cluster1.plot_constraints_accurate(agg_pt, new_pt, plates, new_crystal, k, plot_dots = False)
                #print('agg agg generate rand pt',file=file) 
                #print('here')
                
                agg_pt, new_pt = cluster1.generate_random_point_fast(cluster2, 1)
                movediffx = new_pt.x - agg_pt.x
                movediffy = new_pt.y - agg_pt.y
                cluster2.move([-movediffx, -movediffy, 0])
                
                #print('agg agg closest pts',file=file) 
                nearest_geoms, nearest_geoms_y = cluster1.closest_points(cluster2)
                if nearest_geoms is None and fail == 0:
                    print("cascaded_union FAILED # ", fail)
                    #f6.write('%d\t %.3f\t %d\t\n'%(r, phio, fail))
                    #f6.flush()
                    fail+=1

                cluster1 = cluster1.add_cluster(cluster2)
     
                if rand_orient:
                    cluster1.orient_cluster()#rand_orient =False
                    
                rx,ry,rz = cluster1.spheroid_axes()
                a,b,c = sorted([rx,ry,rz])
       
                #cluster1.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y, 'x')
                #cluster1.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y, 'y')
                #cluster1.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y, 'z')

            
                if l==0 and m==0:

                    rxs[n,l+m]=a
                    rys[n,l+m]=b
                    rzs[n,l+m]=c
                    try:
                        cplxs[n,l+m]=cluster1.complexity

                    except AttributeError: 
                        
                        pass
                    #phi2Ds[n,l+m] = cluster1.phi_2D()

                if m==1:
  
                    rxs[n,l+m]=a
                    rys[n,l+m]=b
                    rzs[n,l+m]=c
                    try:
                        cplxs[n,l+m]=cluster1.complexity

                    except AttributeError: 
                       
                        pass
                    #phi2Ds[n,l+m] = cluster1.phi_2D()
                        
                if l == ncrystals1-1 and m !=0:

                    rxs[n,l+m]=a
                    rys[n,l+m]=b
                    rzs[n,l+m]=c
                    try:
                        cplxs[n,l+m]=cluster1.complexity

                    except AttributeError: 
                        
                        pass
                    #phi2Ds[n,l+m] = cluster1.phi_2D()

                cluster1.remove_cluster(cluster2)

                #print('n,l+m',n, l+m, cluster1.ncrystals,cluster2.ncrystals)
                m+=1  #increment # cluster counter for array indices
                count2 +=1
            l+=1
            count1 +=1

    print('made it to the end of collect_clusters loops')

    f5.close()


    return [phio, r, width, length, rxs, rys, rzs, phi2Ds, cplxs]
  