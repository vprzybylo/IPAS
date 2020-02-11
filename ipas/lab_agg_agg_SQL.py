"""Runs aggregate-aggregate collection."""
import ipas
import time
import numpy as np
import random
import copy

def collect_clusters(clusters, rand_orient=False):
    start_time = time.time()
    array_size = np.sum(range(len(clusters)))
    
    #NEW AGGREGATE PROPERTIES
    cplxs = np.zeros(array_size, dtype=np.float64)
    rxs = np.zeros(array_size, dtype=np.float64)
    rys = np.zeros(array_size, dtype=np.float64)
    rzs = np.zeros(array_size, dtype=np.float64)
    phi2Ds = np.zeros(array_size, dtype=np.float64)  
    dd = np.zeros(array_size, dtype=np.float64)  
    
    '''----START AGG-AGG COLLECTION ------'''
    c=1
    ct = 0
    
    while c <= len(clusters):
        for n in range(len(clusters)-c):
            #print('n', c, n, len(clusters)-c)
            cluster1 = clusters[n]
            cluster2 = clusters[n+c] 
           
            
            if rand_orient:
                cluster1.orient_cluster(rand_orient=True) 
                cluster2.orient_cluster(rand_orient=True)
            else:
                cluster1.orient_cluster() 
                cluster2.orient_cluster()

            agg_pt, new_pt = cluster1.generate_random_point_fast(cluster2, 1)
            movediffx = new_pt.x - agg_pt.x
            movediffy = new_pt.y - agg_pt.y
            cluster2.move([-movediffx, -movediffy, 0])

            fail=0
            nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy = cluster1.closest_points(cluster2)
            if nearest_geoms_xz is None or nearest_geoms_yz is None and fail == 0:
                print("cascaded_union FAILED # ", fail)
                fail+=1
            
            cluster1_hold = copy.deepcopy(cluster1)
            cluster3 = cluster1.add_cluster(cluster2)
            
            if rand_orient:
                cluster3.orient_cluster(rand_orient=True) #rand_orient =False
            else:
                cluster3.orient_cluster()

            rx,ry,rz = cluster3.spheroid_axes(cluster1.plates)         
            rxs[ct],rys[ct],rzs[ct] = sorted([rx,ry,rz])

            #FOR DENSITY CHANGE ------------------
            #monomer a and c
            a1=np.power((np.power(cluster1.monor,3)/cluster1.monophi),(1./3.))
            c1= cluster1.monophi*a1
            a2=np.power((np.power(cluster2.monor,3)/cluster2.monophi),(1./3.))
            c2= cluster2.monophi*a2
            Va1 = 3*(np.sqrt(3)/2) * np.power(c1,2) * a1 * cluster1_hold.ncrystals
            Va2 = 3*(np.sqrt(3)/2) * np.power(c2,2) * a2 * cluster2.ncrystals
            Va3 = np.sum(Va1+Va2) #new volume of agg 

            Ve1 = 4./3.*cluster1.a*cluster1.b*cluster1.c #volume of ellipsoid for clus1
            Ve2 = 4./3.*cluster2.a*cluster2.b*cluster2.c #volume of ellipsoid for clus2
            Ve3 = 4./3.*rx*ry*rz  #volume of ellipsoid for new agg
            #print('Va1, Ve1, Va2, Ve2', Va1, Ve1, Va2, Ve2)
            Vr1 = Va1/Ve1 
            Vr2 = Va2/Ve2
            Vr_avg = np.average([Vr1,Vr2])
            Vr3 = Va3/Ve3  #volumetric ratio after adding an agg, not a monomer
            #print('Vr1, Vr2, Vr3, Vravg', Vr1, Vr2, Vr3, Vr_avg)
            dd[ct] = Vr3-Vr_avg
            #print('dd = ', dd[ct])
            #-------------------------------------

            cluster3.recenter()
            cplxs[ct], circle=cluster3.complexity(cluster1, cluster2)
            #print('after moving')
            #print('cplx', cplxs[ct])
            #print('cluster3 ncrystals', cluster1_hold.ncrystals, cluster2.ncrystals, cluster3.ncrystals)
#             cluster3.plot_ellipsoid_aggs([cluster1_hold, cluster2], nearest_geoms_xz, \
#                                               nearest_geoms_yz, nearest_geoms_xy, view='z', circle=circle)
#             cluster3.plot_ellipsoid_aggs([cluster1_hold, cluster2], nearest_geoms_xz, \
#                                               nearest_geoms_yz, nearest_geoms_xy, view='x', circle=circle)
#             cluster3.plot_ellipsoid_aggs([cluster1_hold, cluster2], nearest_geoms_xz, \
#                                               nearest_geoms_yz, nearest_geoms_xy, view='y', circle=circle)
            phi2Ds[ct] = cluster3.phi_2D()
            ct+=1
        c+=1
    
    print('made it to the end of collect_clusters loops')
    
    #returns arrays of len(# of collections)
    #characteristic values determined in postprocessing
    return rxs, rys, rzs, phi2Ds, cplxs, dd
  