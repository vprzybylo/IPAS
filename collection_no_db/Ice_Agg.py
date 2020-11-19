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
constraint='mpi_ib',
queue='batch',
walltime='04-23:00:00',
cores=1,
memory='4000MiB', #1 GiB = 1,024 MiB
processes=1)
print('dashboard link: ', cluster.dashboard_link)
cluster.scale(50)
client = Client(cluster)
print(client)
print('scheduler info: ',client.scheduler_info())
time.sleep(30)

def main():
    output = np.empty((len(phioarr),len(reqarr), nclusters),dtype=object)
    for phi in range(len(phioarr)):
        for r in range(len(reqarr)):
            for n in range(nclusters):
                output[phi,r, n] = dask.delayed(ipas.collect_clusters)(phioarr[phi], reqarr[r], ncrystals, rand_orient)
            #ipas.collect_clusters(phioarr[phi], reqarr[r], nclusters, ncrystals,rand_orient)
#     delayeds = client.compute(delayeds)
#     output = client.gather(delayeds)
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
    #print(np.shape(gather))
    agg_as = gather[:,:,:,0]
    agg_bs = gather[:,:,:,1]
    agg_cs = gather[:,:,:,2]
    phi2Ds = gather[:,:,:,3]
    phi2D = gather[:,:,:,4]
    cplxs = gather[:,:,:,5] 
    dds = gather[:,:,:,6]
    perims = gather[:,:,:,7]
    print('DONE!')
    return agg_as, agg_bs, agg_cs, phi2Ds, phi2D, cplxs, dds, perims

if __name__ == '__main__':
    #phioarr=np.logspace(-2, 2, num=20, dtype=None)#just columns (0,2); plates (-2,0)
    phioarr = [0.01, 0.1, 1.0, 10., 100.0]#[0.01, 0.10, 0.50, 1.0]

    #reqarr = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
    reqarr = [1000]
    nclusters = 300         #changes how many aggregates per aspect ratio to consider
    ncrystals = 100
    rand_orient = False   #randomly orient the seed crystal and new crystal: uses first random orientation
    
    output = main()
    agg_as, agg_bs, agg_cs, phi2Ds, phi2D, cplxs, dds, perims = compute()
    results = {'agg_as': agg_as, 'agg_bs':agg_bs, 'agg_cs':agg_cs, 'phi2Ds':phi2Ds, \
               'phi2D':phi2D, 'cplxs':cplxs, 'dds':dds, 'perims':perims}

with open('../instance_files/instance_iceagg_flat_n100_a1000_allphi_eqmajorax', "wb") as f:
    pickle.dump(results, f)
    f.close()
    print('finished!')

