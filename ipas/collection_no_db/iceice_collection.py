"""
Utilities for running ice particle simulations.
"""

import ipas.collection_no_db.ipas as ipas
import copy as cp
import numpy as np


def collect_clusters_ice_ice(phio, r, nclusters, ncrystals, rand_orient):

    #NEW AGGREGATE PROPERTIES
    cplxs = []
    agg_as = []
    agg_bs = []
    agg_cs = []
    overlap = []
#    phi2Ds = []
#     dds = []
    depths = []
    major_ax_zs = []

    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    #print('mono a', a)
    #print('mono c', c)
    if c < a:
        plates = True
    else:
        plates = False

    count = 0
    for n in range(nclusters):
        #if n % 20 == 0.:
            #print('nclus',int(np.round(r)), phio, n)

        crystal1 = ipas.IceCrystal(a, c)
        crystal1.hold_clus = crystal1.points
        crystal1.orient_crystal(rand_orient)
        crystal1.recenter()
        #cluster will start with a random orientation if crystal was reoriented
        cluster = ipas.ClusterCalculations(crystal1)  

        while cluster.ncrystals < ncrystals: 

            crystal2 = ipas.IceCrystal(a, c)
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
            a_crys_ell,b_crys_ell,c_crys_ell= crystal1.ellipsoid_axes()
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

            #depths.append(cluster.depth())
            #major_ax_zs.append(cluster.major_ax('z'))
            agg_a, agg_b, agg_c = cluster.ellipsoid_axes() 
            agg_as.append(agg_a)
            agg_bs.append(agg_b)
            agg_cs.append(agg_c)

            #FOR DENSITY CHANGE ------------------
#             Va = 3*(np.sqrt(3)/2) * np.power(a1,2) * c1 *cluster.ncrystals #2
#             Ve = 4./3.*np.pi*cluster.agg_a*cluster.agg_b*cluster.agg_c 
#             d2 = Va/Ve 
#             dd=(d2-d1)/d1
#             dds.append(dd) #change in density
            #print(d1, d2, dd)

            #-------------------------------------
            #cluster.complexity()
            #cplx, circle= cluster.complexity()

            #cplxs.append(cplx)
            #phi2Ds.append(cluster.phi_2D())
            cluster.points = cluster.orient_points

#             print('x')
#             cluster.plot_ellipse([['x','y']])
#             cluster.plot_ellipsoid_aggs([cluster], view='x', circle=None)
#             print('y')
#             cluster.plot_ellipsoid_aggs([cluster], view='y', circle=None)
#             print('z')
#             cluster.plot_ellipsoid_aggs([cluster], view='z', circle=None)

            cluster_cp = cp.deepcopy(cluster)

    #print('made it to the end of collect_clusters loops')
    return agg_as, agg_bs, agg_cs