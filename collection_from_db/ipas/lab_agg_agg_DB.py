"""Runs aggregate-aggregate collection."""
import ipas
import time
import numpy as np
import random
import copy

def collect_clusters(clusters1, clusters2, rand_orient=False):
    
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
    
    for n in range(len(clusters1)):
        cluster1 = clusters1[n]
        cluster2 = clusters2[n] 
        
        if rand_orient:
            cluster1.orient_cluster(rand_orient=True) 
            cluster2.orient_cluster(rand_orient=True)
        else:
            cluster1.orient_cluster() 
            cluster2.orient_cluster()
        cluster1.recenter()
        cluster2.recenter()
        
        agg_pt, new_pt = cluster1.generate_random_point_fast(cluster2, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        cluster2.move([-movediffx, -movediffy, 0])

        nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy = cluster1.closest_points(cluster2)           

        cluster1cp = copy.deepcopy(cluster1)
        cluster3 = cluster1cp.add_cluster(cluster2)
        cluster3.add_points = copy.deepcopy(cluster3.points)

        if rand_orient:
            cluster3.orient_cluster(rand_orient=True) #rand_orient =False
        cluster3.recenter()
        cluster3.orient_points = copy.deepcopy(cluster3.points)
        
        agg_a,agg_b,agg_c = cluster3.spheroid_axes()    
        agg_as.append(agg_a)
        agg_bs.append(agg_b)
        agg_cs.append(agg_c)
        
        #if agg_b/agg_a < 0.4:
#         print('agg mono_phi', cluster1.monophi)
#         print('agg mono r', cluster1.monor)
#         print('agg_b', agg_b)
#         print('agg_a', agg_a)
#         print('b/a ', agg_b/agg_a)

        #DENSITY CHANGE ------------------
        #monomer a and c
        a1=np.power((np.power(cluster1.monor,3)/cluster1.monophi),(1./3.))
        c1= cluster1.monophi*a1
        a2=np.power((np.power(cluster2.monor,3)/cluster2.monophi),(1./3.))
        c2= cluster2.monophi*a2
        Va1 = 3*(np.sqrt(3)/2) * np.power(a1,2) * c1 * cluster1.ncrystals
        Va2 = 3*(np.sqrt(3)/2) * np.power(a2,2) * c2 * cluster2.ncrystals
        Va3 = np.sum(Va1+Va2) #new volume of agg 

        Ve1 = 4./3.*np.pi*cluster1.a*cluster1.b*cluster1.c #volume of ellipsoid for clus1
        Ve2 = 4./3.*np.pi*cluster2.a*cluster2.b*cluster2.c #volume of ellipsoid for clus2
        Ve3 = 4./3.*np.pi*agg_a*agg_b*agg_c  #volume of ellipsoid for new agg
        #print('Va1, Ve1, Va2, Ve2', Va1, Ve1, Va2, Ve2)
        Vr1 = Va1/Ve1 
        Vr2 = Va2/Ve2
        Vr_avg = np.average([Vr1,Vr2])
        Vr3 = Va3/Ve3  #volumetric ratio after adding an agg, not a monomer
        #print('Vr1, Vr2, Vr3, Vravg', Vr1, Vr2, Vr3, Vr_avg)
        
        dds.append((Vr3-Vr_avg)/Vr_avg)
        #print('dd', (Vr3-Vr_avg)/Vr_avg)
        #-------------------------------------

        cplx, circle= cluster3.complexity()
        cplxs.append(cplx)
        phi2Ds.append(cluster3.phi_2D_rotate())
        cluster3.points = cluster3.orient_points
        
        #print('cluster3 ncrystals', cluster1.ncrystals, cluster2.ncrystals, cluster3.ncrystals)

#         print('w')   
        #if agg_b/agg_a < 0.4:
        #cluster3.plot_ellipsoid_aggs([cluster1, cluster2], view='w', circle=None)
#         cluster3.plot_ellipsoid_aggs([cluster1, cluster2], view='y', circle=None)
        cluster3.plot_ellipsoid_aggs([cluster1, cluster2], view='z', circle=None)
        #cplx, circle= cluster2.complexity()
#         cluster2.plot_ellipsoid_aggs([cluster2], view='z', circle=None)
#         cluster3.plot_ellipsoid_aggs([cluster1, cluster2], view='z', circle=None)
        
    print('made it to the end of collect_clusters loops')
    
    #returns arrays of len(# of collections)
    #characteristic values determined in postprocessing
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds
  