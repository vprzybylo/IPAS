"""Runs aggregate-aggregate collection."""
import ipas
import time
import numpy as np
import random
import copy as cp

def collect_clusters(a, c, clusters, rand_orient=False):
    
    
    #NEW AGGREGATE PROPERTIES
#     cplxs = np.zeros(array_size, dtype=np.float64)
#     rxs = np.zeros(array_size, dtype=np.float64)
#     rys = np.zeros(array_size, dtype=np.float64)
#     rzs = np.zeros(array_size, dtype=np.float64)
#     phi2Ds = np.zeros(array_size, dtype=np.float64)  
#     dd = np.zeros(array_size, dtype=np.float64) 
    cplxs = []
    agg_as = []
    agg_bs = []
    agg_cs = []
    phi2Ds = []    
    dds = []
    
    for n in range(len(clusters)):
        monomer = ipas.Ice_Crystal(a[n], c[n])
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
        
        #for density change
        rx,ry,rz = cluster.spheroid_axes()  
        Ve_clus = 4./3.*np.pi*rx*ry*rz 
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

        agg_a, agg_b, agg_c = cluster.spheroid_axes()  
 
        #DENSITY CHANGE ------------------
        
        Ve3 = 4./3.*np.pi*agg_a*agg_b*agg_c  #volume of ellipsoid for new agg
        d2 = (Va_clus+Va_mono)/Ve3
        
        #print((d2-d1)/d1)
        dds.append((d2-d1)/d1)

        #-------------------------------------

        cplx, circle= cluster.complexity()
        cplxs.append(cplx)
        phi2Ds.append(cluster.phi_2D_rotate())
        cluster.points = cluster.orient_points
        if monomer.phi < 1.0:
            cluster.plot_ellipsoid_aggs([cluster, monomer], view='x', circle=circle, agg_agg=False)
            cluster.plot_ellipsoid_aggs([cluster, monomer], view='z', circle=circle, agg_agg=False)

#        cluster.plot_ellipsoid_aggs([cluster, monomer], view='y', circle=None, agg_agg=False)
#        cluster.plot_ellipsoid_aggs([cluster, monomer], view='z', circle=None, agg_agg=False)
    #characteristic values determined in postprocessing
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds
  