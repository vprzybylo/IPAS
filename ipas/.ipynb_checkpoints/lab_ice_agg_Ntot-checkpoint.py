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

from ipas import plots_phiarr as plts
from ipas import IceClusterBatchIceAgg_Ntot as batch
from ipas import IceCrystal as crys
from ipas import IceClusterAggAgg as clus

# def profile(func):
#      def wrapper(*args, **kwargs):
#          start = time.time()
#          func(*args, **kwargs)
#          end   = time.time()
#          print(end - start)
#      return wrapper



def orient_crystal(length, width, rand_orient=False):
    #create and orient a crystal and return it to be collected
    
    if rand_orient:
        # make a new crystal, orient it
        crystal = crys.IceCrystal(length=length, width=width)
        crystal.reorient()

    else:
        
        f = lambda x: -crys.IceCrystal(length=length, width=width, rotation=[x,0,0]).projectxy().area
        xrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
        f = lambda x: -crys.IceCrystal(length=length, width=width, rotation=[0,x,0]).projectxy().area
        yrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
        #xrotrand = random.uniform(-xrot, xrot)
        #yrotrand = random.uniform(-yrot, yrot)
        zrot=random.uniform(0, 2 * np.pi)
        xrotrand = random.uniform(-xrot, xrot)
        yrotrand = random.uniform(-yrot, yrot)

        if width > length:
            #yrot=random.uniform(0, 2 * np.pi)
            best_rotation = [xrotrand, yrotrand, zrot]
        else:
            best_rotation = [0,yrotrand,zrot]


        crystal = crys.IceCrystal(length=length, width=width, rotation=best_rotation)

    return crystal

def orient_cluster(cluster, length, width, rand_orient=False):
     
    if rand_orient:
        cluster.reorient()

    else:
        
        f = lambda x: -cluster.rotate_to([x,0,0]).projectxy().area
        xrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
        f = lambda x: -cluster.rotate_to([0,x,0]).projectxy().area
        yrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
        zrot=random.uniform(0, 2 * np.pi)
        xrotrand = random.uniform(-xrot, xrot)
        yrotrand = random.uniform(-yrot, yrot)
        best_rot = [xrot, yrot, zrot]

        best_rotation = [xrot, yrot, 0]
        cluster.rotate_to(best_rot)

        #cluster.plot_ellipse([['x','y']])
        #cplxs[n,l+m] = cluster.complexity()

    return(cluster)


def collect_clusters(phio, r, nclusters, ncrystals, save_plots=False, numaspectratios=50, 
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
    
    cplxs = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    rxs = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    rys = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    rzs = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
    phi2Ds = np.zeros((nclusters,(ncrystals)-1), dtype=np.float64)
   
    
    width = (r**3/phio)**(1./3.)
    length=phio*width
    #all monomers the same for now
    r = np.power((np.power(width,2)*length),(1./3.)) #of monomer
    
    f5 = open('outfile_allflat_rall.dat',"w")       

    k=0  #for saving plot constraints
    
    for n in range(nclusters):
        
        if n % 50 == 0:
            #print('nclus',r, phio, n)
            f5.write('%d\t %.3f\t %d\t\n'%(r, phio, n))
            f5.flush()
       
        crystal = orient_crystal(length, width, rand_orient)
        cluster1 = clus.IceCluster(crystal)

        l=0  #crystals in cluster1
        fail=0
        while cluster1.ncrystals < ncrystals:  #agg 1

            #start_ice = time.time()

            new_crystal = orient_crystal(length, width, rand_orient)

            agg_pt, new_pt = cluster1.generate_random_point(new_crystal, 1)
            movediffx = new_pt.x - agg_pt.x
            movediffy = new_pt.y - agg_pt.y
            new_crystal.move([-movediffx, -movediffy, 0])

            #cluster1.closest_points(new_crystal)
            cluster1.add_crystal_from_above(new_crystal, lodge=lodge)
            #cluster1.plot_ellipsoid()
            cluster1._add_crystal(new_crystal)   
            
            if rand_orient:
                cluster1 = orient_cluster(cluster1, length, width, rand_orient=False)
                
            #cluster1.plot_ellipsoid()
            Va = 3*(np.sqrt(3)/2) * np.power(width,2) * length * cluster1.ncrystals  #actual agg volume of hexagonal prisms
            rx, ry, rz = cluster1.spheroid_axes()  #radii lengths - 3 axes
            a,b,c = sorted([rx,ry,rz])
                    
            cplxs[n,l]=cluster1.complexity()
            rxs[n,l]=a
            rys[n,l]=b
            rzs[n,l]=c
            phi2Ds[n,l] = cluster1.phi_2D()
            l+=1
        
    print('made it to the end of collect_clusters loops')
    f5.close()
   

    return [phio, r, width, length, rxs, rys, rzs, phi2Ds, cplxs]
  