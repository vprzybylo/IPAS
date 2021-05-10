"""
Main function for running ice particle simulations
ICE-ICE collection
"""

import ipas.collection_no_db.crystal as crys
import ipas.collection_no_db.calculations as clus
import copy as cp
import numpy as np


def collect_clusters_iceice(phio, r, ncrystals,
                            rand_orient, plot=False):

    #NEW AGGREGATE PROPERTIES
    agg_as = np.empty(ncrystals-1)
    agg_bs = np.empty(ncrystals-1)
    agg_cs = np.empty(ncrystals-1)
    cplxs = np.empty(ncrystals-1)
    phi2Ds = np.empty(ncrystals-1)
    dds = np.empty(ncrystals-1)

    # a and c axes of monomer using phi and r
    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    if c < a:
        plates = True
    else:
        plates = False
    
    # create Crystal
    crystal1 = crys.Crystal(a, c)
    crystal1.hold_clus = crystal1.points
    crystal1.orient_crystal(rand_orient)
    crystal1.recenter()

    # create cluster from initialized crystal
    # same orientation if crystal was reoriented
    cluster = clus.ClusterCalculations(crystal1)  

    l=0
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
        A = cluster.fit_ellipsoid()
        cluster.ellipsoid_axes_lengths(A)
        agg_as[l] = cluster.a
        agg_bs[l] = cluster.b
        agg_cs[l] = cluster.c

        # --------------------------
        # volume of all monomers in cluster
        Va = 3*(np.sqrt(3)/2) * np.power(a1,2) * c1 *cluster.ncrystals
        # volume of ellipsoid around cluster after aggregation
        Ve = 4./3.*np.pi*cluster.a*cluster.b*cluster.c 
        d2 = Va/Ve 
        dd = (d2-d1)/d1
        dds[l] = dd #change in density

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
        cplxs[l] = cplx
        phi2Ds[l] = cluster.phi_2D()
        # reset points back to how they were before phi_2D_rotate
        cluster.points = cluster.orient_points

        # -------- PLOTTING --------
        if plot:
            print('x')
            cluster.plot_ellipsoid_aggs([cluster], view='x', circle=None)
            print('y')
            cluster.plot_ellipsoid_aggs([cluster], view='y', circle=None)
            print('z')
            cluster.plot_ellipsoid_aggs([cluster], view='z', circle=None)

        cluster_cp = cp.deepcopy(cluster)

    #print('made it to the end of collect_clusters loops')
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds