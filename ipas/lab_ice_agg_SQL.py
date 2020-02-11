"""Utilities for running ice particle simulations."""
import copy as cp
import numpy as np
import ipas 
#from .db import Session
import time
import logging
import multiprocessing
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, event, select
from sqlite3 import Connection as SQLite3Connection

def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        #cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.close()

    
def collect_clusters(phio, notebook, r, nclusters, ncrystals, rand_orient=False, lodge=0):
    print('rand in collect = ', rand_orient)
    engine = create_engine('sqlite:///IPAS_%d_400_%.3f_lastmono.sqlite' %(notebook, phio))
    event.listen(engine, 'connect', _set_sqlite_pragma)
    ipas.base.Base.metadata.create_all(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    cplxs = np.zeros((nclusters, ncrystals - 1), dtype=np.float64)
    rxs = np.zeros((nclusters, ncrystals - 1), dtype=np.float64)
    rys = np.zeros((nclusters, ncrystals - 1), dtype=np.float64)
    rzs = np.zeros((nclusters, ncrystals - 1), dtype=np.float64)
    phi2Ds = np.zeros((nclusters, ncrystals - 1), dtype=np.float64)
    list_of_clusters = []
    list_of_crystals = []

    width = (r ** 3 / phio) ** (1. / 3.)
    length = phio * width
    if length > width:
        plates = False
    else:
        plates = True

    # all monomers the same for now

    r = np.power((np.power(width, 2) * length), (1. / 3.))  # of monomer
    #outfile = open('outfile_allrand_rall.dat', "a+")

    count = 0
    for n in range(nclusters):
        if n % 20 == 0.:
            print('nclus',int(np.round(r)), phio, n, float(time.time())/60)
            #outfile.write('%.0f\t %.3f\t %d\t\n' % (r, phio, n))
            #outfile.flush()

        crystal = ipas.IceCrystal(length=length, width=width, rand_orient=rand_orient)
        crystal.orient_crystal(rand_orient)

        cluster1 = ipas.IceCluster(crystal)
        cluster1.mono_phi = phio
        cluster1.ncrystals = 1

        l = 0  # crystals in cluster1
        
        while cluster1.ncrystals < ncrystals:  # agg 1
            xmax = cluster1.points['x'].max()
            ymax = cluster1.points['y'].max()
            xmin = cluster1.points['x'].min()
            ymin = cluster1.points['y'].min()
            rand_center = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax), 0]

            new_crystal = ipas.IceCrystal(length=length, width=width, center= rand_center, rand_orient=rand_orient)
            new_crystal.orient_crystal(rand_orient)

            #ck_intersect = False
            #while ck_intersect is False:
                #agg_pt, new_pt = cluster1.generate_random_point(new_crystal, 1)
                #movediffx = new_pt.x - agg_pt.x
                #movediffy = new_pt.y - agg_pt.y
                #new_crystal.move([-movediffx, -movediffy, 0])

                #'''NEW NEAREST PT METHOD for ice-agg'''
                # cluster2.closest_points(new_crystal)
                # ck_intersect = True
                # print('clus 2 add xtal from above',file=file)


            ck_intersect = cluster1.add_crystal_from_above(new_crystal, lodge=lodge)
            miss = 0
            while ck_intersect is False:
                rand_center = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax), 0]
                new_crystal = ipas.IceCrystal(length=length, width=width, center=rand_center, rand_orient=rand_orient)
                new_crystal.orient_crystal(rand_orient)
                ck_intersect = cluster1.add_crystal_from_above(new_crystal, lodge=lodge)
                miss+=1
                if miss >1000:
                    print('uh oh')
            cluster1.add_crystal(new_crystal)
            
            if rand_orient:
                cluster1.orient_cluster()  # rand_orient =False

            # cluster1.plot_ellipsoid()
            Va = 3 * (np.sqrt(3) / 2) * np.power(width,2) * length * cluster1.ncrystals  # actual agg volume of hexagonal prisms
            a, b, c = cluster1.spheroid_axes(plates)  # radii lengths - 3 axes

            cplxs[n, l] = cluster1.complexity()
            rxs[n, l] = a
            rys[n, l] = b
            rzs[n, l] = c
            phi2Ds[n, l] = cluster1.phi_2D_rotate()

            cluster1cp = cp.deepcopy(cluster1)
            #print(new_crystal.points)
            cluster1cp.crystal.append(new_crystal)
            #new_crystal.aggs.append(cluster1cp)
            
            #list_of_clusters.append(cluster1cp)
            try:
                session.add(cluster1cp)
                session.commit()
            except:
                print('in except')
                raise
     
            count += 1
            l += 1

#     try:
#         session.add_all(list_of_clusters)  # crystal id has been appended into cluster relationship
#         session.commit()
        
#     except:
#         print('in except')
#         raise        
    session.close() 
    print('done committing phio = %.3f %.3f' %(phio, float(time.time())/60))

    
#     # ADD TO DATABASE
#     try:
#         session = Session()
#         start = time.time()
#         session.add_all(list_of_clusters)  # crystal id has been appended into cluster relationship
#         session.commit()
#         end = time.time()
#         time_taken = end - start
#         print(time_taken)
        
        
#         #session.close()
#     except:
#         print('in except')
#         raise

    #print('made it to the end of collect_clusters loops')
    #outfile.close()
    
    return 
    #return list_of_clusters
    #return [phio, r, width, length, rxs, rys, rzs, phi2Ds, cplxs]
