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

from ipas import plots_phiarr as plts
from ipas import IceClusterBatchAggAgg_Ntot as batch
from ipas import IceCrystal as crys
from ipas import IceClusterAggAgg as clus

# def profile(func):
#      def wrapper(*args, **kwargs):
#          start = time.time()
#          func(*args, **kwargs)
#          end   = time.time()
#          print(end - start)
#      return wrapper

def parallelize_clusters(phio, reqarr, save_plots=False,
                        minor='minor_axis',nclusters=300, ncrystals1=10,
                        ncrystals2=10, numaspectratios=50,
                        rand_orient=False, ch_dist='gamma'):
    chreq = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    chphi = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    chphi2D = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ovrlp = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    S = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ch_ovrlp = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ch_S = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ch_majorax = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ch_minorax = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ch_cplx = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    dphigam = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_phi = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_phi = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_phi2D = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_phi2D = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_mjrax = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_mjrax = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_req = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_req = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_minor_axis = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_minor_axis = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_ovrlp = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_ovrlp = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_S = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_S = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_cplx = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_cplx = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    min_phi = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    max_phi = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    min_phi2D = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    max_phi2D = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    min_mjrax = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    max_mjrax = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    min_minor_axis = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    max_minor_axis = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    min_req = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    max_req = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_phi = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_phi2D = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_ovrlp = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_S = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_mjrax = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_minor_axis = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_req = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_cplx = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    poserr_d2 = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    negerr_d2 = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ch_d2 = np.ones(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)), dtype=np.float64)
    min_d2 = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    max_d2 = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    mean_d2 = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)
    ch_dd = np.zeros(shape=(len(phio),len(reqarr),(ncrystals1+ncrystals2)-1), dtype=np.float64)

    find_N = np.zeros(shape=(len(phio),len(reqarr),1), dtype=np.float64)
    f1 = open('find_N_lookup.dat',"w+")
    f2 = open('major_axis_lookup.dat',"w+")
    f3 = open('minor_axis_lookup.dat',"w+")
    f4 = open('dd_lookup.dat',"w+")

    phioarr = []
    xrot = []
    yrot = []
    widtharr = []
    lengtharr = []

    seed_phi_ind = 0
    start_time = time.time()
    for i in range(len(phio)):
        seed_req_ind = 0
        for s in range(len(reqarr)):   #array of equivalent volume radii to loop through
            #equivalent volume length and width for each aspect ratio
            #r**3  volume
            width = (reqarr[s]**3/phio[i])**(1./3.)
            length=phio[i]*width
            #all monomers the same for now
            r = np.power((np.power(width,2)*length),(1./3.)) #of monomer

            print('eq. vol rad', r, length, width, phio[i])

            #cProfile.runctx('sim_clusters(r, length=length, width=width, nclusters=nclusters, ncrystals=ncrystals, numaspectratios = numaspectratios, rand_orient=rand_orient)', None,locals(), 'out.profile')

            b1 = collect_clusters(r, length=length, width=width, nclusters=nclusters, ncrystals1=ncrystals1,
                                  ncrystals2=ncrystals2, numaspectratios = numaspectratios,
                                  rand_orient=rand_orient)

            widtharr.append(width)
            lengtharr.append(length)
            phioarr.append(phio)

            #After clusters are made, pass each variable name to return the characteristic
            #of the given variable distribution of n clusters.

            b1.get_characteristics(var ='phi', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            poserr_phi[i,s,:]=b1.poserr[:]
            negerr_phi[i,s,:]=b1.negerr[:]
            chphi[i,s,:]=b1.ch[:]
            min_phi[i,s,:]=b1.min_data[:]
            max_phi[i,s,:]=b1.max_data[:]
            mean_phi[i,s,:]=b1.mean[:]
            #print(chphi[i,s,:])
            #print('done with phi')

            b1.get_characteristics(var ='req', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            poserr_req[i,s,:]=b1.poserr[:]
            negerr_req[i,s,:]=b1.negerr[:]
            chreq[i,s,:]=b1.ch[:]
            min_req[i,s,:]=b1.min_data[:]
            max_req[i,s,:]=b1.max_data[:]
            mean_req[i,s,:]=b1.mean[:]
            #print('done with req')

            b1.get_characteristics(var ='density_change', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            poserr_d2[i,s,:]=b1.poserr[:]
            negerr_d2[i,s,:]=b1.negerr[:]
            ch_d2[i,s,:-1]=b1.ch[:]
            min_d2[i,s,:]=b1.min_data[:]
            max_d2[i,s,:]=b1.max_data[:]
            mean_d2[i,s,:]=b1.mean[:]
            ch_d2[i,s,:]= np.insert(ch_d2[i,s,:-1], 0, 1.0)
            #print('d1', ch_d2[i,s,:-1])
            #print('d2', ch_d2[i,s,1:])
            ch_dd[i,s,:] = (ch_d2[i,s,:-1] - ch_d2[i,s,1:])
            #print('after denstiy subtraction', ch_dd[i,s,:])

            b1.get_characteristics(var ='major_axis', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            poserr_mjrax[i,s,:]=b1.poserr[:]
            negerr_mjrax[i,s,:]=b1.negerr[:]
            ch_majorax[i,s,:]= b1.ch[:]
            min_mjrax[i,s,:]=b1.min_data[:]
            max_mjrax[i,s,:]=b1.max_data[:]
            mean_mjrax[i,s,:]=b1.mean[:]
            #print('done with major axis')

            b1.get_characteristics(var ='minor_axis', save=save_plots, minor = minor, ch_dist=ch_dist, verbose=False)
            poserr_minor_axis[i,s,:]=b1.poserr[:]
            negerr_minor_axis[i,s,:]=b1.negerr[:]
            ch_minorax[i,s,:]=b1.ch[:]
            min_minor_axis[i,s,:]=b1.min_data[:]
            max_minor_axis[i,s,:]=b1.max_data[:]
            mean_minor_axis[i,s,:]=b1.mean[:]
            #print('done with minor axis')


            #write out lookup tables
            

            for N in range(len(chphi[i,s,:])):
                #to find N

                f1.write('%.3f\t %d\t %.4f\t %.4f\t\n'%(phio[i], reqarr[s], chphi[i,s,N], chreq[i,s,N]))
                f1.flush()
                print(phio[i],reqarr[s],chphi[i,s,N], chreq[i,s,N])
                #major_axis
                f2.write('%.3f\t %d\t %.4f\t\n'%(phio[i], reqarr[s], ch_majorax[i,s,N]))
                print(phio[i], reqarr[s], ch_majorax[i,s,N])
                f2.flush()
		#minor_axis
                f3.write('%.3f\t %d\t %.4f\t\n'%(phio[i], reqarr[s], ch_minorax[i,s,N]))
                f3.flush()
		#density Change
                f4.write('%.3f\t %d\t %.4f\t\n'%(phio[i], reqarr[s], ch_dd[i,s,N]))
                f4.flush()

            print("--- %.2f minute(s) ---" % ((time.time() - start_time)/60))
            print('made it to the end of parallelize_clusters')

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    
    return (phioarr, widtharr, lengtharr, chphi, ch_majorax, ch_minorax, poserr_phi, negerr_phi,
               min_phi, max_phi, mean_phi, mean_mjrax, mean_minor_axis, poserr_d2, negerr_d2,
               min_d2, max_d2, ch_dd)


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

    if rand_orient or cluster.ncrystals>=20:
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


def collect_clusters(r, length, width, nclusters, ncrystals1, ncrystals2, rand_orient=False,
                numaspectratios=50, minor='minor_axis', lodge=0, max_misses=20):

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
            'random'. <-- should leave as is ('IDL' was for testing purposes).
        minor (str): The minor axis measurement used in calculating aggregate aspect
            ratio. 'minor_axis' for the max-min point parallel to the z axis.
            'minorxy' for the minor axis length from the fit ellipse in either the
            x or y orientation (whichever is longer).  Default is 'minor_axis'.
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

    cplxs = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    phi = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    major_axis = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    minor_axis = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    req = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    real_phi = np.zeros((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)
    d2 = np.ones((nclusters,(ncrystals1+ncrystals2)-1), dtype=np.float64)

    #tiltdiffsx = []
    #tiltdiffsy = []
    #print('n', nclusters, ncrystals1*ncrystals2)

    k=0  #for saving plot constraints
    for n in range(nclusters):

        if n % 50 == 0:
            print('nclus',n)
        #print('nclus',n)
        plates = width > length

        # initialize cluster  1 and 2 with 1 xtal

        crystal = orient_crystal(length, width, rand_orient)
        cluster1 = clus.IceCluster(crystal)
        count1 = 0
        count2 = 0
        l=0  #crystals in cluster1
        while cluster1.ncrystals < ncrystals1:  #agg 1

            #start_ice = time.time()
            #print('n,l+m',cluster1.ncrystals,cluster2.ncrystals, ncrystals1)

            if count1 > 0:

                if cluster1.ncrystals == 1:
                    l=1

                '''Generating random points to combine'''

                new_crystal = orient_crystal(length, width, rand_orient)
                #cluster1.plot_ellipsoid()
                #if rand_orient:
                #    rand_orient = False
                #    cluster1 = orient_cluster(cluster1, length, width, rand_orient)
                #    rand_orient = True
                #cluster1 = orient_cluster(cluster1, length, width, rand_orient)
                agg_pt, new_pt = cluster1.generate_random_point(new_crystal, 1)
                movediffx = new_pt.x - agg_pt.x
                movediffy = new_pt.y - agg_pt.y
                new_crystal.move([-movediffx, -movediffy, 0])

                '''NEW NEAREST PT METHOD for ice-agg'''
                #cluster1.closest_points(new_crystal)
                cluster1.add_crystal_from_above(new_crystal, lodge=lodge)
                #cluster1.plot_ellipsoid()
                cluster1._add_crystal(new_crystal)

                #cluster.plot_constraints_accurate(agg_pt, new_pt, plates, new_crystal, k, plot_dots = False)
                '''
                while nmisses < max_misses:
                    crystal_hit = cluster1.add_crystal_from_above(new_crystal, lodge=lodge)
                    #crystal_hit = cluster1.check_intersection(new_crystal)
                    if crystal_hit:
                        cluster1 = ice_agg_collection(cluster1, new_crystal, rand_orient)
                        break
                    else:
                        print('xtal false hit')
                        nmisses += 1
                        if nmisses > max_misses:
                            print('crystal hit miss max %d > %d' %(nmisses, max_misses))
                            break
                '''
            crystal = orient_crystal(length, width, rand_orient)
            cluster2 = clus.IceCluster(crystal)

            #cluster2.plot_ellipsoid_agg_agg(cluster2)
            m=0  #crystals in cluster2
            while cluster2.ncrystals < ncrystals2:  #agg 2

                if cluster2.ncrystals == 1:
                    m=1

                if count2 > 0:
                    new_crystal = orient_crystal(length, width, rand_orient)
                    if rand_orient:
                        rand_orient = False
                        cluster2 = orient_cluster(cluster2, length, width, rand_orient)
                        rand_orient = True

                    agg_pt, new_pt = cluster2.generate_random_point(new_crystal, 1)
                    movediffx = new_pt.x - agg_pt.x
                    movediffy = new_pt.y - agg_pt.y
                    new_crystal.move([-movediffx, -movediffy, 0])

                    '''NEW NEAREST PT METHOD for ice-agg'''
                    #cluster2.closest_points(new_crystal)
                    cluster2.add_crystal_from_above(new_crystal, lodge=lodge)
                    cluster2._add_crystal(new_crystal)
                    #cluster2.plot_ellipsoid()

                else:
                    m=0  #to start index for dd at 0 for 1 crystal (ice-ice)

                '''----AGG-AGG COLLECTION ------'''
                #start_agg = time.time()
                #cluster.plot_constraints_accurate(agg_pt, new_pt, plates, new_crystal, k, plot_dots = False)
                agg_pt, new_pt = cluster1.generate_random_point(cluster2, 1)
                movediffx = new_pt.x - agg_pt.x
                movediffy = new_pt.y - agg_pt.y
                cluster2.move([-movediffx, -movediffy, 0])

                nearest_geoms, nearest_geoms_y = cluster1.closest_points(cluster2)
                #cluster1.plot_ellipsoid()

                cluster1._add_cluster(cluster2)

                #print('before cluster reorientation')
                #cluster1.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y,view='x')
                #cluster1.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y,view='y')

                if rand_orient:
                    cluster1 = orient_cluster(cluster1, length, width, rand_orient=False)

                #print('after cluster reorientation')

                #cluster1.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y,view = 'x')
                #cluster1.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y,view = 'y')
                #cluster2.plot_ellipsoid()

                rx, ry, rz = cluster1.spheroid_axes()

                #calculate density
                if l==0 and m==0:

                    Va = 3*np.sqrt(3)/2 * np.power(width,2) * length * cluster1.ncrystals  #actual agg volume of hexagonal prisms
                    rx, ry, rz = cluster1.spheroid_axes()  #radii lengths - 3 axes
                    #print(rx, ry, rz)
                    Ve = 4/3*rx*ry*rz  #equiv volume density from fit ellipsoid
                    #an equivalent ratio of densities - close to 1.0 for single monomer, otherwise <1.0
                    d2[n,l+m]=Va/Ve

                    minor_axis[n,l+m] = min(rx,ry,rz)
                    major_axis[n,l+m] = max(rx,ry,rz)

                    if plates:
                        req[n,l+m] = np.power((np.power(major_axis[n,l+m],2)*minor_axis[n,l+m]),(1./3.))
                        phi[n,l+m]=(minor_axis[n,l+m])/(major_axis[n,l+m])

                    else:
                        req[n,l+m] = np.power((np.power(minor_axis[n,l+m],2)*major_axis[n,l+m]),(1./3.))
                        phi[n,l+m]=(major_axis[n,l+m])/(minor_axis[n,l+m])

                if m==1:

                    Va = 3*np.sqrt(3)/2 * np.power(width,2) * length * cluster1.ncrystals  #actual agg volume of hexagonal prisms
                    rx, ry, rz = cluster1.spheroid_axes()  #radii lengths - 3 axes
                    #print(rx, ry, rz)
                    Ve = 4/3*rx*ry*rz  #equiv volume density from fit ellipsoid
                    d2[n,l+m]=Va/Ve
                    #print('dd check', l+m)

                    minor_axis[n,l+m] = min(rx,ry,rz)
                    major_axis[n,l+m] = max(rx,ry,rz)

                    if plates:
                        req[n,l+m] = np.power((np.power(major_axis[n,l+m],2)*minor_axis[n,l+m]),(1./3.))
                        phi[n,l+m]=(minor_axis[n,l+m])/(major_axis[n,l+m])

                    else:
                        req[n,l+m] = np.power((np.power(minor_axis[n,l+m],2)*major_axis[n,l+m]),(1./3.))
                        phi[n,l+m]=(major_axis[n,l+m])/(minor_axis[n,l+m])

                if l == ncrystals1-1 and m !=0:

                    Va = 3*np.sqrt(3)/2 * np.power(width,2) * length * cluster1.ncrystals  #actual agg volume of hexagonal prisms
                    rx, ry, rz = cluster1.spheroid_axes()  #radii lengths - 3 axes
                    #print(rx, ry, rz)
                    Ve = 4/3*rx*ry*rz  #equiv volume density from fit ellipsoid
                    d2[n,l+m]=Va/Ve
                    #print('dd check', l+m)

                    minor_axis[n,l+m] = min(rx,ry,rz)
                    major_axis[n,l+m] = max(rx,ry,rz)

                    if plates:
                        req[n,l+m] = np.power((np.power(major_axis[n,l+m],2)*minor_axis[n,l+m]),(1./3.))
                        phi[n,l+m]=(minor_axis[n,l+m])/(major_axis[n,l+m])

                    else:
                        req[n,l+m] = np.power((np.power(minor_axis[n,l+m],2)*major_axis[n,l+m]),(1./3.))
                        phi[n,l+m]=(major_axis[n,l+m])/(minor_axis[n,l+m])
                #print('cluster1.ncrystals', cluster1.ncrystals)
                cluster1._remove_cluster(cluster2)

                #print('n,l+m',n, l+m, cluster1.ncrystals,cluster2.ncrystals)
                m+=1  #increment # cluster counter for array indices
                count2 +=1
            l+=1
            count1 +=1

    print('made it to the end of collect_clusters')

    return batch.IceClusterBatch(ncrystals1, ncrystals2, length, width, numaspectratios,
                                 cplxs, phi, major_axis, minor_axis, req, d2, real_phi, plates)
