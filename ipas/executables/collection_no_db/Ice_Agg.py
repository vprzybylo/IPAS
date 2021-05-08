import ipas
import matplotlib.pyplot as plt
import numpy as np
import time
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress
import dask
from scipy import spatial 
from shapely.geometry import Point
from shapely.ops import nearest_points
import pandas as pd
from dask import dataframe as dd
import pickle

cluster = SLURMCluster(
queue='batch',
walltime='04-23:00:00',
cores=1,
memory='10000MiB', #1 GiB = 1,024 MiB
processes=1)
print('dashboard link: ', cluster.dashboard_link)
cluster.scale(28*2)
client = Client(cluster)
print(client)
print('scheduler info: ',client.scheduler_info())
time.sleep(30)

def main():
    output = np.empty((len(phioarr),len(reqarr), nclusters),dtype=object)
    for phi in range(len(phioarr)):
        for r in range(len(reqarr)):
            for n in range(nclusters):
                output[phi,r, n] = dask.delayed(ipas.collect_clusters_alldask)(phioarr[phi],
                                                                               reqarr[r],
                                                                               ncrystals,
                                                                               rand_orient)
            #ipas.collect_clusters_alldask(phioarr[phi], reqarr[r], nclusters, ncrystals,rand_orient)
    delayeds = client.compute(delayeds)
    output = client.gather(delayeds)
    return output

def compute():
    agg_as = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    agg_bs = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    agg_cs = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    phi2Ds = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    phi2D = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    cplxs = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    dds = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    perims = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))

    gather = client.compute([*output.tolist()])  #only parallelizing agg r bins
    gather = client.gather(gather)

    gather = np.array(gather)
    print(np.shape(gather))
    agg_as = gather[:,:,:,0]
    agg_bs = gather[:,:,:,1]
    agg_cs = gather[:,:,:,2]
    phi2D = gather[:,:,:,3]
    dds = gather[:,:,:,4] 

    print('DONE!')
    return agg_as, agg_bs, agg_cs, phi2D, dds

if __name__ == '__main__':

    # monomer aspect ratios (all the same in agg)
    phioarr = [0.1, 0.5, 1.0, 2.0, 10.0]
    # monomer radii 
    reqarr = [10]
    # how many aggregates to produce
    nclusters = 100
    # how many monomers per aggregate
    ncrystals = 150
    # orientation of the monomers
    rand_orient = False

    output = main()
    agg_as, agg_bs, agg_cs, phi2D, dds = compute()

    results = {'agg_as': agg_as, 'agg_bs':agg_bs, 'agg_cs':agg_cs, \
               'phi2D':phi2D, 'dds':dds}

    with open('../instance_files/instance_iceagg_flat_n150_a10_phi5_eqmajorax', "wb") as f:
        pickle.dump(results, f)
        f.close()
        print('finished!') 

