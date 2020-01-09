"""Runs aggregate-aggregate collection."""
import ipas
import time
import numpy as np
import random

def collect_clusters(clusters, rand_orient=False):
    
    #NEW AGGREGATE PROPERTIES
    cplxs = np.zeros((len(clusters), dtype=np.float64)
    rxs = np.zeros((len(clusters), dtype=np.float64)
    rys = np.zeros((len(clusters), dtype=np.float64)
    rzs = np.zeros((len(clusters), dtype=np.float64)
    phi2Ds = np.zeros((len(clusters), dtype=np.float64)  
    
    '''----START AGG-AGG COLLECTION ------'''
                      
    for n in range(len(clusters)):
        #randomly choose out of n clusters which two to aggregate
        cluster1 = ipas.IceCluster(random.choice(clusters))
        cluster2 = ipas.IceCluster(random.choice(clusters))
        agg_pt, new_pt = cluster1.generate_random_point_fast(cluster2, 1)
        movediffx = new_pt.x - agg_pt.x
        movediffy = new_pt.y - agg_pt.y
        cluster2.move([-movediffx, -movediffy, 0])

        fail=0
        nearest_geoms, nearest_geoms_y = cluster1.closest_points(cluster2)
        if nearest_geoms is None and fail == 0:
            print("cascaded_union FAILED # ", fail)
            fail+=1

        cluster1 = cluster1.add_cluster(cluster2)

        if rand_orient:
            cluster1.orient_cluster() #rand_orient =False

        rx,ry,rz = cluster1.spheroid_axes()
        a,b,c = sorted([rx,ry,rz])

        #ADD DENSITY CHANGE

        rxs[n]=a
        rys[n]=b
        rzs[n]=c
        cplxs[n]=cluster1.complexity()
        phi2Ds[n] = cluster1.phi_2D()

    print('made it to the end of collect_clusters loops')

    return [rxs, rys, rzs, phi2Ds, cplxs]
  