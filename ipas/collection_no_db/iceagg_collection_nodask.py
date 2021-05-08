"""
Main function for running ice particle simulations
ICE-AGG collection
includes looping over ncrystals instead of doing that outside in a dask computation
"""

from ipas.collection_no_db.crystal import Crystal
from ipas.collection_no_db.calculations import ClusterCalculations
import copy as cp
import numpy as np


def collect_clusters_iceagg(phio, r, nclusters, ncrystals, rand_orient):

    # NEW AGGREGATE PROPERTIES
    cplxs = np.empty((nclusters, ncrystals-1))
    agg_as = np.empty((nclusters, ncrystals-1))
    agg_bs = np.empty((nclusters, ncrystals-1))
    agg_cs = np.empty((nclusters, ncrystals-1))
    phi2Ds = np.empty((nclusters, ncrystals-1)) 
    dds = np.empty((nclusters, ncrystals-1))

    # get a and c axes of monomer using phi and r
    a = (r ** 3 / phio) ** (1. / 3.)
    c = phio * a
    if c < a:
        plates = True
    else:
        plates = False

    # how many aggregates to create
    for n in range(nclusters):
        # create Crystal
        crystal1 = Crystal(a, c)
        crystal1.hold_clus = crystal1.points
        crystal1.orient_crystal(rand_orient)
        crystal1.recenter()

        # create cluster from initialized crystal
        # same orientation if crystal was reoriented
        cluster = ClusterCalculations(crystal1)

        l=0
        # number of monomers/crystals per aggregate/cluster
        while cluster.ncrystals < ncrystals: 
            # initialize a new crystal
            crystal2 = Crystal(a,c)
            crystal2.hold_clus = crystal2.points
            crystal2.orient_crystal(rand_orient)
            crystal2.recenter()

            # move monomer on top of cluster
            agg_pt, new_pt = cluster.generate_random_point_fast(crystal2, 1)
            movediffx = new_pt.x - agg_pt.x
            movediffy = new_pt.y - agg_pt.y
            crystal2.move([-movediffx, -movediffy, 0])

            # move cluster and monomer together
            cluster.closest_points(crystal2)

            # ----------- DENSITY CHANGE ----------
            # get cluster ellipsoid axes before aggregation
            rx,ry,rz = cluster.ellipsoid_axes()  
            # volume of ellipsoid around cluster before aggregation
            Ve_clus = 4./3.*np.pi*rx*ry*rz 

            # a and c of monomers in cluster (all identical)
            a_clus=np.power((np.power(cluster.mono_r,3)/cluster.mono_phi),(1./3.))
            c_clus = cluster.mono_phi*a_clus
            # volume of all monomers in cluster
            Va_clus = 3*(np.sqrt(3)/2) * np.power(a_clus,2) * c_clus * cluster.ncrystals
            # density ratio of aggregate and ellipsoid
            d1 = Va_clus/Ve_clus

            # -------------------
            # add monomer points to original cluster (i.e., aggregate)
            cluster.add_crystal(crystal2)
            # save original points before reorienting for max area
            cluster.add_points = cp.deepcopy(cluster.points)
            # -------------------

            # monomer a and c axes
            a_mono = np.power((np.power(crystal2.r,3)/crystal2.phi),(1./3.))
            c_mono = crystal2.phi*a_mono
            # volume of monomer to collect
            Va_mono = 3*(np.sqrt(3)/2) * np.power(a_mono,2) * c_mono

            # get fit-ellipsoid radii (a-major, c-minor) after aggregation
            agg_a, agg_b, agg_c = cluster.ellipsoid_axes()  
            agg_as[n,l] = agg_a
            agg_bs[n,l] = agg_b
            agg_cs[n,l] = agg_c

            # volume of ellipsoid around cluster after aggregation
            Ve3 = 4./3.*np.pi*agg_a*agg_b*agg_c  #volume of ellipsoid for new agg
            d2 = (Va_clus+Va_mono)/Ve3
            # append relative change in density (after - before adding monomer)
            dds[n,l] = (d2-d1)/d1

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
            cplxs[n,l], circle= cluster.complexity()
            phi2Ds[n,l] = cluster.phi_2D()
            # reset points back to how they were before phi_2D_rotate
            cluster.points = cluster.orient_points

            # -------- PLOTTING --------
#             print('w')
#             cluster.plot_ellipsoid_aggs([cluster, crystal2], view='w', circle=None, agg_agg=False)
#             print('x')
#             cluster.plot_ellipsoid_aggs([cluster, crystal2], view='x', circle=None, agg_agg=False)
#             print('y')
#             cluster.plot_ellipsoid_aggs([cluster, crystal2], view='y', circle=None, agg_agg=False)
#             print('z')
#             cluster.plot_ellipsoid_aggs([cluster, crystal2], view='z', circle=None, agg_agg=False)

            cluster_cp = cp.deepcopy(cluster)
            l+=1

    # characteristic values determined in postprocessing
    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds