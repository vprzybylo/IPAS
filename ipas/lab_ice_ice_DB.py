"""Runs aggregate-aggregate collection."""
import ipas
import time
import numpy as np
import random
import copy

def collect_clusters(monomers1, monomers2, rand_orient=False):
    
    #NEW AGGREGATE PROPERTIES
    cplxs = []
    rxs = []
    rys = []
    rzs = []
    phi2Ds = []    
    dd = []
    
    '''----START ICE-AGG COLLECTION ------'''

    for n in range(len(monomers1)):
        cluster1 = monomers1[n]
        cluster2 = monomers2[n]
        
        cluster1.recenter()
        cluster2.recenter()
        
#         cluster1.plot_ellipsoid(cluster1, view='x')
#         cluster1.plot_ellipsoid(cluster1, view='y')
#         #cluster1._rotate_to([0,0,90]) 
        
        if rand_orient:
            cluster1.orient_crystal(rand_orient=True) 
            cluster2.orient_crystal(rand_orient=True)
        else:
            cluster1.orient_crystal() 
            cluster2.orient_crystal()
            
#         cluster1.recenter()
#         cluster2.recenter()
  
#         cluster1.plot_ellipsoid(cluster1, view='x')
#         cluster1.plot_ellipsoid(cluster1, view='y')
        
#         cluster2.plot_ellipsoid(cluster2, view='x')
#         cluster2.plot_ellipsoid(cluster2, view='y')

        agg_pt, new_pt = cluster1.generate_random_point_fast(cluster2, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        cluster2.move([-movediffx, -movediffy, 0])
        
        nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy = cluster1.closest_points(cluster2)
        
        #this has to go after the last movement of crystal2    
        cluster1cp = copy.deepcopy(cluster1)
        cluster3 = cluster1cp.add_crystal(cluster2)
        cluster3 = ipas.Cluster_Calculations(cluster3)
        cluster3.ncrystals += 1

        if rand_orient:
            cluster3.orient_cluster(rand_orient=True) #rand_orient =False
        else:
            cluster3.orient_cluster()

        rx,ry,rz = cluster3.spheroid_axes()  
        rx, ry, rz = sorted([rx,ry,rz])
        rxs.append(rx)
        rys.append(ry)
        rzs.append(rz)

        #FOR DENSITY CHANGE ------------------
        #monomer a and c
        a1=np.power((np.power(cluster1.r,3)/cluster1.phi),(1./3.))
        c1= cluster1.phi*a1
        a2=np.power((np.power(cluster2.r,3)/cluster2.phi),(1./3.))
        c2= cluster2.phi*a2
        Va1 = 3*(np.sqrt(3)/2) * np.power(c1,2) * a1 * cluster1.ncrystals
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
        dd.append(Vr3-Vr_avg)
        #-------------------------------------

        cluster3.recenter()
        cplx, circle= cluster3.complexity(cluster1, cluster2)
        cplxs.append(cplx)
        #print('after moving')
        
        print('x')
        cluster3.plot_ellipsoid_aggs([cluster1, cluster2], view='x', circle=None)
        print('y')
        cluster3.plot_ellipsoid_aggs([cluster1, cluster2], view='y', circle=None)
        print('z')
        cluster3.plot_ellipsoid_aggs([cluster1, cluster2], view='z', circle=None)

        phi2Ds.append(cluster3.phi_2D())

    #characteristic values determined in postprocessing
    print('len of data: ', len(rxs), len(dd))
    return rxs, rys, rzs, phi2Ds, cplxs, dd
  