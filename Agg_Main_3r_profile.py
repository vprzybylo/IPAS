from ipas import lab_agg_agg_Ntot_1loop as lab
from ipas import IceCrystal as crys
from ipas import IceClusterAggAgg as clus
from ipas import plots_phiarr as plts
import numpy as np
import pandas as pd
import time
import itertools   #to create width array and join plate/col aspect ratios
from operator import itemgetter
import shapely.geometry as geom
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random
import warnings
import sys


phioarr=np.logspace(2, 2, num=1, dtype=None)#just columns (0,2); plates (-2,0)
lenphio = len(phioarr)

reqarr = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
reqarr = [10]
numaspectratios=len(phioarr)
ch_dist='gamma'         #anything other than gamma uses the characteristic from the best distribution pdf (lowest SSE)
nclusters = 50        #changes how many aggregates per aspect ratio to consider
ncrystals1 = 20       #number of monomers per aggregate 1
ncrystals2 = 20          #number of monomers per aggregate 2
minor = 'depth'        #'minorxy' from fit ellipse or 'depth' to mimic IPAS in IDL
rand_orient = False    #randomly orient the seed crystal and new crystal: uses first random orientation
save_plots = False     #saves all histograms to path with # of aggs and minor/depth folder
file_ext = 'eps'


def main():
    start = time.clock()
    for phio in phioarr:
        for r in reqarr:
            b1 = lab.collect_clusters(phio, r, nclusters=nclusters,
                                ncrystals1=ncrystals1, ncrystals2=ncrystals2, 
                                save_plots=save_plots, numaspectratios=numaspectratios,
                                rand_orient=rand_orient, lodge=0, max_misses=20,
                                ch_dist=ch_dist)
    end = time.clock()
    print("%.s" % (end-start))
    return b1



if __name__ == '__main__':

    main()

