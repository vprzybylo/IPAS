import ipas.collection_no_db.iceice_collection as collect
import numpy as np
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask
import pickle
import argparse
import time
import sys

def start_client(num_workers):
    '''
    initialize dask client
    '''
    cluster = SLURMCluster(
    queue='batch',
    walltime='04-23:00:00',
    cores=1,
    memory='10000MiB', #1 GiB = 1,024 MiB
    processes=1)
    print('dashboard link: ', cluster.dashboard_link)
    cluster.scale(num_workers)
    client = Client(cluster)
    print(client)
    print('scheduler info: ',client.scheduler_info())
    time.sleep(5)
    
    return client


def write_file(filename, agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds):
    '''write results to file''' 
    results = {'agg_as': agg_as, 'agg_bs':agg_bs, 'agg_cs':agg_cs,
              'phi2Ds': phi2Ds,' cplxs': cplxs, 'dds': dds}
    print('saving results to ', filename)
    filehandler = open(filename, 'wb')
    pickle.dump(results, filehandler)
    filehandler.close()
    print('done writing!')


def compute(phioarr, reqarr, nclusters, ncrystals,
            rand_orient, use_dask=False, num_workers=2,
            plot=False):
    '''
    collect monomers and return aggregate attributes
    '''
    
    agg_as = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    agg_bs = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    agg_cs = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    phi2Ds = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    cplxs = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    dds = np.empty((len(phioarr),len(reqarr), nclusters, ncrystals-1))
    
    output = np.empty((len(phioarr),len(reqarr), nclusters),dtype=object)
    for phi in range(len(phioarr)):
        for r in range(len(reqarr)):
            for n in range(nclusters):
                if use_dask:
                    output[phi,r,n] = dask.delayed(collect.collect_clusters_iceice)(phioarr[phi], reqarr[r],
                                                                                   ncrystals, rand_orient, plot=plot)
                else:
                    agg_as[phi,r,n], agg_bs[phi,r,n], agg_cs[phi,r,n], phi2Ds[phi,r,n], cplxs[phi,r,n], dds[phi,r,n] = collect.collect_clusters_iceice(phioarr[phi], reqarr[r], ncrystals,
                                                                                                    rand_orient, plot=plot)
    
    if use_dask:
        #start dask client
        client = start_client(num_workers)
        gather = client.compute([*output.tolist()]) 
        gather = client.gather(gather)
        gather = np.array(gather)
        agg_as = gather[:,:,:,0,:]
        agg_bs = gather[:,:,:,1,:]
        agg_cs = gather[:,:,:,2,:]
        phi2Ds = gather[:,:,:,3,:]
        cplxs = gather[:,:,:,4,:] 
        dds = gather[:,:,:,5,:]

    return agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds


def main():
    
    parser = argparse.ArgumentParser(description='ice-ice collection using IPAS')
    parser.add_argument('--phi', '-p', nargs="+", default=[0.1, 1.0, 10.0],
                        metavar='aspect ratio', required=True, type=float,
                        help='monomer aspect ratio(s)')
    parser.add_argument('--radius', '-r', metavar='radius',
                        nargs="+", default=[100], required=True,
                        help='monomer radius/radii', type=int)
    parser.add_argument('--aggregates', '-a', metavar='number of aggregates',
                        required=True, type=int,
                        help='number of aggregates to create per aspect ratio-radius pairing')
    parser.add_argument('--monomers', '-m', metavar='number of monomers',
                        required=False, default=2,
                        help='number of monomers per aggregate', type=int)
    parser.add_argument('--rand', metavar='orientation',
                        help='monomer and aggregate orientation',
                        type=bool, default=False)
    parser.add_argument('--save', '-s', metavar='save aggregate attributes', 
                        help='save aggregate attributes to pickled file',
                        type=bool, default=False)
    parser.add_argument('--filename', '-f', metavar='saving filename',
                        help='filename to save data (include path from execution location)',
                        type=str, default=False, required='--filename' in sys.argv)
    parser.add_argument('--use_dask', metavar='parallelization',
                        help='use dask client for parallelization',
                        type=bool, default=False)
    parser.add_argument('--workers', '-w', metavar='workers',
                        help='number of workers for dask client',
                        type=int, default=5, required='--use_dask' in sys.argv)
    
        
    args = parser.parse_args()
    
    
    # monomer aspect ratios (all the same in agg)
    phioarr = args.phi
    # monomer radii 
    reqarr = args.radius
     # how many aggregates to produce
    nclusters = args.aggregates
    # number of monomers per aggregate
    ncrystals = args.monomers
    # monomer orientation - random (True) or flat (False)
    rand_orient = args.rand
    print('creating aggregates..')
    save = args.save
    if save:
        filename = args.filename
    use_dask = args.use_dask
    if use_dask:
        num_workers = args.workers
        agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds = compute(phioarr, reqarr,
                                                             nclusters, ncrystals,
                                                             rand_orient, use_dask=use_dask,
                                                             num_workers=num_workers)

    else:
        agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds = compute(phioarr, reqarr, nclusters,
                                                             ncrystals, rand_orient)

    if save:
        write_file(filename, agg_as, agg_bs, agg_cs, phi2Ds, cplxs, dds)
    
    print('finished!')
    
if __name__ == '__main__':    
    main()
    
