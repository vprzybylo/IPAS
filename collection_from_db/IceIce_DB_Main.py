import ipas 
import numpy as np
import dask
from dask_jobqueue import SLURMCluster
from distributed import LocalCluster
from dask.distributed import Client, progress
from dask import delayed
from dask import dataframe as dd
import functools
import sys
import ast
from struct import *
import pickle
import glob
import random
import pandas as pd
import time
from dask.distributed import as_completed
from joblib import Parallel, delayed, parallel_backend
import matplotlib.pyplot as plt

cluster = SLURMCluster(
    queue='kratos',
    walltime='04-23:00:00',
    cores=1,
    memory='20000MiB', #1 GiB = 1,024 MiB
    processes=1)

cluster.scale(20)
time.sleep(100)

client = Client(cluster)
print(client)
print(cluster.scheduler.services['dashboard'].server.address)
print('dashboard port:', cluster.scheduler.services['dashboard'].server.port)

files = ['sqlite:///'+f for f in glob.glob("db_files/IPAS_*_flat.sqlite")]
tables = ['aggregates', 'crystals']

df=[]
for table in tables:
    
    #read tables in parallel on client 
    read_files = [dask.delayed(dd.read_sql_table)(table=table, uri=file, index_col='id') for file in files]
    
    compute_read = client.compute(read_files)
    print('done with compute')
    ddfs = client.gather(compute_read)
    print('done with gather')
    #concatenate all sqlite files vertically (axis=0 default) (same columns)
    gathered_reads = client.scatter(ddfs)
    ddf = client.submit(dd.concat, gathered_reads).result()
    print('done with submit')
    #append combined dask df for each table
    df.append(ddf)
    
df_concat = dd.concat([df[0], df[1]], axis=1)
df_concat.agg_r = np.power((np.power(df_concat.a, 2) * df_concat.c), (1./3.))
def query_r_5000(df):
    return df[df.agg_r < 5000]

df_concat = df_concat.map_partitions(query_r_5000)
#len(df_concat) #86% of dataset
df_repart = df_concat.repartition(partition_size="100MB").persist()
df_repart.npartitions

def main():
    
    output = np.empty((mono_phi_bins,mono_r_bins),dtype=object)
    hold_monos1 = np.empty((mono_phi_bins,mono_r_bins,nclusters), dtype=object)
    hold_monos2 = np.empty((mono_phi_bins,mono_r_bins,nclusters), dtype=object)
    
    for j in range(mono_phi_bins): 
        for k in range(mono_r_bins):   
            print('phio, r', phio_m[j],req_m[k])
            
            df_mono = df[1][(df[1].phi < phio_m[j]+(phio_m[j]*0.01)) &\
                            (df[1].phi > phio_m[j]-(phio_m[j]*0.01)) &\
                            (df[1].r < req_m[k]+(req_m[k]*0.01)) &\
                            (df[1].r > req_m[k]-(req_m[k]*0.01))].compute()

            samples_mono = df_mono.sample(nclusters)
            
            n_monos=0
            for mono in samples_mono.itertuples():
                mono = ipas.IceCrystal(mono)
                
                hold_monos1[j,k,n_monos] = mono
                n_monos+=1
                
            samples_mono = df_mono.sample(nclusters)
            
            n_monos=0
            for mono in samples_mono.itertuples():
                
                mono = ipas.IceCrystal(mono)
                
                hold_monos2[j,k,n_monos] = mono
                n_monos+=1
      
            #ipas.collect_clusters(hold_monos1[j,k,:], hold_monos2[j,k,:], rand_orient=rand_orient)
            output[j,k] = dask.delayed(ipas.collect_clusters)(hold_monos1[j,k,:],\
                                                                  hold_monos2[j,k,:], rand_orient=rand_orient)
    print('done with appending collect clusters!')
    return output
    
def compute():
    results = np.empty((mono_phi_bins, mono_r_bins, nclusters))
    rxs = np.empty((mono_phi_bins, mono_r_bins, nclusters))
    rys = np.empty((mono_phi_bins, mono_r_bins, nclusters))
    rzs = np.empty((mono_phi_bins, mono_r_bins, nclusters))
    phi2Ds = np.empty((mono_phi_bins, mono_r_bins, nclusters))
    cplxs = np.empty((mono_phi_bins, mono_r_bins, nclusters))
    dd = np.empty((mono_phi_bins, mono_r_bins, nclusters))

    gather = client.compute([*output.tolist()])  #only parallelizing agg r bins
    gather = client.gather(gather)

    gather = np.array(gather)
    print(np.shape(gather))
    rxs = gather[:,:,0,:]
    rys = gather[:,:,1,:]
    rzs = gather[:,:,2,:]
    phi2Ds = gather[:,:,3,:]
    cplxs = gather[:,:,4,:] 
    dd = gather[:,:,5,:]

    print('DONE with compute!')
    return rxs, rys, rzs, phi2Ds, cplxs, dd

if __name__ == '__main__':
    rand_orient = False      #randomly orient the seed crystal and new crystal: uses first random orientation
    req_m = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]

    mono_phi_bins = 20
    mono_r_bins = len(req_m)
    nclusters = 300

    phio_m=np.logspace(-2, 2, num=mono_phi_bins, dtype=None)#just columns (0,2); plates (-2,0)

    output = main()
    rxs, rys, rzs, phi2Ds, cplxs, dd = compute()
    results = {'rxs': rxs, 'rys':rys, 'rzs':rzs, 'phi2Ds':phi2Ds, \
               'cplxs':cplxs, 'dd':dd}

    
    filename = 'instance_files/instance_db_iceice_flat'
    filehandler = open(filename, 'wb')
    pickle.dump(results, filehandler)
    filehandler.close()
    
