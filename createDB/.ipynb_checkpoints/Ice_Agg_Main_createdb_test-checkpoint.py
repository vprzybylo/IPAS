import ipas
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial
import time
from multiprocessing import Process
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress
import dask

#import threading

phioarr=np.logspace(-2, 2, num=10, dtype=None)#just columns (0,2); plates (-2,0)
lenphio = len(phioarr)

reqarr = [300]
numaspectratios=len(phioarr)
ch_dist='gamma'         #anything other than gamma uses the characteristic from the best distribution pdf (lowest SSE)
nclusters = 30        #changes how many aggregates per aspect ratio to consider
ncrystals = 50       #number of monomers per aggregate 1
minor = 'depth'        #'minorxy' from fit ellipse or 'depth' to mimic IPAS in IDL
rand_orient = True    #randomly orient the seed crystal and new crystal: uses first random orientation
save_plots = False     #saves all histograms to path with # of aggs and minor/depth folder
file_ext = 'eps'

def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA synchronous=0")
        cursor.execute("PRAGMA jounal_mode=MEMORY")
        #cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.close()


def main():
    cluster = SLURMCluster(
    queue='kratos',
    walltime='04-23:00:00',
    cores=1,
    memory='20000MiB', #1 GiB = 1,024 MiB
    processes=1)

    cluster.scale(10)
    client = Client(cluster)
    time.sleep(30)
    print(client)
    
    for r in reqarr:
        for phi in phioarr:
            parallel_clus=partial(ipas.collect_clusters, r, nclusters,ncrystals,rand_orient)
            delayeds.append(dask.delayed(parallel_clus)(phi))
    print(delayeds)
    delayeds = client.compute(delayeds)
    client.gather(delayeds)
    client.close()


if __name__ == '__main__':
    engine = create_engine('sqlite:///../db_files_04_2020/%.2f_%.3f_%s.sqlite' %(r, phio, rand_orient))
    event.listen(engine, 'connect', _set_sqlite_pragma)
    ipas.base.Base.metadata.create_all(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    main()
