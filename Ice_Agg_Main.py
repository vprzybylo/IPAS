import ipas
import matplotlib.pyplot as plt
import numpy as np
from ipas import lab_ice_agg_SQL as lab
from multiprocessing import Pool
from functools import partial
import time
from multiprocessing import Process
#import threading

phioarr=np.logspace(-2, 2, num=20, dtype=None)#just columns (0,2); plates (-2,0)
lenphio = len(phioarr)

reqarr = [400]
numaspectratios=len(phioarr)
ch_dist='gamma'         #anything other than gamma uses the characteristic from the best distribution pdf (lowest SSE)
nclusters = 300        #changes how many aggregates per aspect ratio to consider
ncrystals = 50       #number of monomers per aggregate 1
minor = 'depth'        #'minorxy' from fit ellipse or 'depth' to mimic IPAS in IDL
rand_orient = True    #randomly orient the seed crystal and new crystal: uses first random orientation
save_plots = False     #saves all histograms to path with # of aggs and minor/depth folder
file_ext = 'eps'

def main():
	notebook=5
	print('notebook = ', notebook)
	print('rand orient = ', rand_orient)
#	pool = Pool(processes=20) #pools are reusable

	for r in reqarr:
		print('r = ',r)
		count = 1
#        pool = Pool(processes=20) #pools are reusable
		parallel_clus=partial(lab.collect_clusters, notebook=notebook, r=r, nclusters=nclusters,\
					ncrystals=ncrystals,rand_orient=rand_orient)
	
		start = time.time()
			
		processes = [Process(target=parallel_clus, args=(phi,)) for phi in phioarr]
		for p in processes:		
			p.start()
		for p in processes:
			p.join()
#		for done in output:
#            		print("after %3.1fsec: count:%s"%(time.time()-start, count))
#            		count +=1

#         for res in output:
#             print("(after %3.1fsec)  mono phi:%.3f  count:%s"%(time.time()-start, res[0].mono_phi, count))
#             count += 1
	print('closing')
	#pool.close()
	#print('joining')
	#pool.join()


if __name__ == '__main__':
	startmain = time.time()
	main()
	endmain = time.time()
	print('time to complete all: ', endmain-startmain)
    
    
