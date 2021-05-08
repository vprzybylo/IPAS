"""
Utilities for running ice particle simulations
"""

import ipas.collection_no_db.crystal as crys
import ipas.collection_no_db.calculations as clus
import copy as cp
import numpy as np


def collect_clusters_ice_ice(phio, r, nclusters, ncrystals, rand_orient):

    #NEW AGGREGATE PROPERTIES
    agg_as = []
    agg_bs = []
    agg_cs = []
    phi2Ds = []
    cplxs = []
    dds = []

    # a and c axes of monomer using phi and r
    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    if c < a:
        plates = True
    else:
        plates = False

    count = 0
    for n in range(nclusters):
        #if n % 20 == 0.:
            #print('nclus',int(np.round(r)), phio, n)

        crystal1 = crys.Crystal(a, c)
        crystal1.hold_clus = crystal1.points
        crystal1.orient_crystal(rand_orient)
        crystal1.recenter()

        # create cluster from initialized crystal
        #cluster will start with a random orientation if crystal was reoriented
        cluster = clus.ClusterCalculations(crystal1)  

        # number of monomers/crystals per aggregate/cluster
        while cluster.ncrystals < ncrystals: 

            crystal2 = crys.Crystal(a, c)
            crystal2.hold_clus = crystal2.points
            crystal2.orient_crystal(rand_orient)
            crystal2.recenter()

            # move monomer on top of cluster
            agg_pt, new_pt = cluster.generate_random_point_fast(crystal2, 1)
            movediffx = new_pt.x - agg_pt.x
            movediffy = new_pt.y - agg_pt.y
            crystal2.move([-movediffx, -movediffy, 0])

            cluster.combine(crystal2)

            # ----------- DENSITY CHANGE ----------
            #starting volume ratio between monomer and fit ellipsoid
            a1=np.power((np.power(crystal1.r,3)/crystal1.phi),(1./3.))
            c1= crystal1.phi*a1
            Va = 3*(np.sqrt(3)/2) * np.power(a1,2) * c1 
            a_crys_ell,b_crys_ell,c_crys_ell= crystal1.ellipsoid_axes()
            Ve = 4./3.*np.pi*a_crys_ell*b_crys_ell*c_crys_ell 
            d1 = Va/Ve 

            cluster.add_crystal(crystal2)
            cluster.add_points = cp.deepcopy(cluster.points)

            #get fit-ellipsoid radii (a-major, c-minor) after aggregation
            agg_a, agg_b, agg_c = cluster.ellipsoid_axes()
            agg_as.append(agg_a)
            agg_bs.append(agg_b)
            agg_cs.append(agg_c)

            # --------------------------
            # volume of all monomers in cluster
            Va = 3*(np.sqrt(3)/2) * np.power(a1,2) * c1 *cluster.ncrystals
            # volume of ellipsoid around cluster after aggregation
            Ve = 4./3.*np.pi*cluster.agg_a*cluster.agg_b*cluster.agg_c 
            d2 = Va/Ve 
            dd=(d2-d1)/d1
            dds.append(dd) #change in density

            # ----------------------------
            # orient cluster after adding monomer
            if a>c and rand_orient== False:
                cluster.orient_cluster() 
            else:
                cluster.orient_cluster(rand_orient) 
            cluster.recenter()

            # ------other calculations------
            # save points before reorienting to calculate phi_2D_rotate
            cluster.orient_points = cp.deepcopy(cluster.points)
            cluster.complexity()
            cplx, circle= cluster.complexity()
            cplxs.append(cplx)
            phi2Ds.append(cluster.phi_2D())
            # reset points back to how they were before phi_2D_rotate
            cluster.points = cluster.orient_points

            # -------- PLOTTING --------
#             print('x')
#             cluster.plot_ellipse([['x','y']])
#             cluster.plot_ellipsoid_aggs([cluster], view='x', circle=None)
#             print('y')
#             cluster.plot_ellipsoid_aggs([cluster], view='y', circle=None)
#             print('z')
#             cluster.plot_ellipsoid_aggs([cluster], view='z', circle=None)

            cluster_cp = cp.deepcopy(cluster)

    #print('made it to the end of collect_clusters loops')
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds