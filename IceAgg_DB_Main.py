# In[2]:
import ipas 
import numpy as np
import dask
from dask_jobqueue import SLURMCluster
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

#cluster.adapt(minimum=3, maximum=20)
cluster.scale(20)

client = Client(cluster)

time.sleep(15) #sec
print(client)
dask.config.set({"distributed.comm.timeouts.tcp": "60s"})


#Initialize databases for queries


files = ['sqlite:///'+f for f in glob.glob("db_files/IPAS_*_lastmono.sqlite")]
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


# # MAIN

ch_dist='gamma'         #anything other than gamma uses the characteristic from the best distribution pdf (lowest SSE)
rand_orient = True  #randomly orient the seed crystal and new crystal: uses first random orientation
save_plots = False     #saves all histograms to path with # of aggs and minor/depth folder

def concatenate_points_all(agg):

    ncrystals = agg.ncrystals    
    #print('ncrystals', ncrystals)
    agg_id = agg.agg_id
    #print('ncrystals, phi, r, agg_id', ncrystals, agg_id, agg_id-ncrystals)
    
    query = df_repart[(df_repart.r == agg.r) & (df_repart.phi == agg.phi) & \
                      (df_repart.ncrystals >= 2) & (df_repart.ncrystals <= ncrystals) & \
                      (df_repart.agg_id <= agg_id) & (df_repart.agg_id >= agg_id-ncrystals)].compute()
    
    cluster = ipas.Cluster_Calculations(agg)
    hold_points = []
    for crys in query.itertuples():
        for points in pickle.loads(crys.points):
            hold_points.append(points)
        #print('hold points', hold_points)

    #cluster.points = np.concatenate(hold_points)
    cluster.points = np.reshape(hold_points, (int(np.shape(hold_points)[0]/12), 12))
    cluster.points = np.array(cluster.points, dtype=[('x', float), ('y', float), ('z', float)])           
#     points = np.concatenate(hold_points)
#     points = np.reshape(points, (int(np.shape(cluster.points)[0]/12), 12))
#     points = np.array(points, dtype=[('x', float), ('y', float), ('z', float)])    
  
    return cluster


def main():

    #req_m = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
    req_m = [1,2,3]

    agg_phi_bins = 2
    agg_r_bins = 2
    mono_phi_bins = 2
    mono_r_bins = len(req_m)
    nclusters = 3


#     agg_phi_bins = 20
#     agg_r_bins = 20
#     mono_phi_bins = 20
#     mono_r_bins = len(req_m)
#     nclusters = 300
    phio_m=np.logspace(-2, 2, num=agg_phi_bins, dtype=None)#just columns (0,2); plates (-2,0)

   
    output = np.empty((mono_phi_bins,mono_r_bins,agg_phi_bins,agg_r_bins,nclusters))
    hold_clusters  = np.empty((agg_phi_bins,agg_r_bins,nclusters), dtype=object)
    hold_monos = np.empty((mono_phi_bins,mono_r_bins,nclusters), dtype=object)
    
    for j in range(mono_phi_bins): 
        for k in range(mono_r_bins):   
            df_mono_phi = df_repart[(df_repart.phi == phio_m[j]) & (df_repart.r == req_m[k])].compute()
            samples_mono = df_mono_phi.sample(nclusters)
            
            n_monos=0
            for mono in samples_mono.itertuples():
                mono = ipas.Cluster_Calculations(mono)
                mono.points = np.array([pickle.loads(mono.points)], dtype=[('x', float), ('y', float), ('z', float)])        
                mono.agg_r = None
                mono.agg_phi = None
                mono.ncrystals = 1
                mono.a = ((mono.r*2)**3./phio_m[j])**(1./3.)
                mono.b = 2*((mono.a/2.)*np.sin(60))
                mono.c = phio_m[j]*mono.a
                hold_monos[j,k,n_monos] = mono
                n_monos+=1
    
            res, phi_bins = pd.qcut(df_repart.agg_phi.compute(), agg_phi_bins, retbins=True)
            print(phi_bins)

            for i in range(agg_phi_bins-1):  #agg phi

                #print('phi_bin = ', phi_bins[i], phi_bins[i+1])
                #return a df that only queries within an aspect ratio bin
                df_phi = df_repart[(df_repart.agg_phi > phi_bins[i]) & (df_repart.agg_phi < phi_bins[i+1]) & \
                                   (df_repart.ncrystals > 2)]  
                #to ensure at least 2 crystals within agg since ncrystals=1 not in db
                #now break that aspect ratio bin into 20 equal r bins
                res, r_bins = pd.qcut(df_phi.agg_r.compute(), agg_r_bins, retbins=True)
                print(r_bins)

                for r in range(agg_r_bins-1):   #agg r
                    print('j, k, i, r ', j, k, i, r)

                    print('r = ', r_bins[r], r_bins[r+1])
                    df_r = df_phi[(df_phi.agg_r > r_bins[r]) & (df_phi.agg_r < r_bins[r+1]) & (df_phi.ncrystals > 2)].compute() 

                    #print(df_repart.id.value_counts().compute().head(30))         

                    start_time = time.time()

                    samples = df_r.sample(nclusters)

                    start_time = time.time()
                    n_aggs=0
                    for agg in samples.itertuples():
                        cluster = concatenate_points_all(agg)
                        hold_clusters[i,r,n_aggs] = cluster
                        n_aggs+=1
                    print('time to concatenate all pts = ', (time.time()-start_time))

                    delayeds = []
                    for agg in samples.itertuples():
                        delayeds.append(dask.delayed(concatenate_points_all)(agg))
                    delayeds = client.compute(delayeds)
                    hold_clusters[i,r,:] = client.gather(delayeds)
                    print('time to concatenate all pts = ', (time.time()-start_time))
                    
                    output[j,k,i,r,:] = dask.delayed(ipas.collect_clusters)(hold_clusters[i,r,:], hold_monos[i,r,:], \
                                                                      rand_orient=rand_orient)

    results = np.empty((mono_phi_bins,mono_r_bins,agg_phi_bins,agg_r_bins))
    start_time = time.time()
    for j in range(mono_phi_bins):
        for k in range(mono_r_bins):
            for i in range(agg_phi_bins):
                gather = client.compute(output[j,k,i,:,:])  #only parallelizing agg r bins
                results[j,k,i,:] = client.gather(gather)

    print('time to collect = ', (time.time()-start_time))
    print('DONE!')
    
    return results, hold_clusters


if __name__ == '__main__':
    results, hold_clusters = main() 
   
    filename = 'instance_files/pulled_clusters_rand_iceagg'
    filehandler = open(filename, 'wb')
    pickle.dump(hold_clusters, filehandler)
    filehandler.close()
    print('finished!')
    
    filename = 'instance_files/instance_db_iceagg_rand'
    filehandler = open(filename, 'wb')
    pickle.dump(output, filehandler)
    filehandler.close()
