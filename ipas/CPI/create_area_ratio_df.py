import glob
import pickle
import sys
from multiprocessing import Pool

import agg_properties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as geom
import shapely.ops as shops
import tables
from dask import dataframe as dd
from pandarallel import pandarallel
from shapely.geometry import Point

import ipas.cluster_calculations as cc

sys.path.append("/network/rit/lab/sulialab/share/IPAS/ipas/collection_from_db")


pandarallel.initialize(progress_bar=False)


# Read Database
orientation = "rand"  # chose which orientation (rand or flat)
if orientation == "rand":
    rand_orient = (
        True
    )  # randomly orient the seed crystal and new crystal: uses first random orientation
    files = glob.glob("../instance_files/createdb_iceagg_rand*")
else:
    rand_orient = (
        False
    )  # randomly orient the seed crystal and new crystal: uses first random orientation
    files = glob.glob("../instance_files/createdb_iceagg_flat*")

db = database.Database(files)
db.read_database()
db.append_shape()
db.truncate_agg_r(5000)
db.append_agg_phi()
df = db.df  # get the dataframe (db is an instance of database.py module)


def shape(a, b, c):
    if (b - c) <= (a - b):
        return "prolate"
    else:
        return "oblate"


df["agg_r"] = np.power((np.power(df["a"], 2) * df["c"]), (1.0 / 3.0))
df = df[df.agg_r < 5000]
# speed up shape function
vfunc = np.vectorize(shape)
df["shape"] = vfunc(df["a"], df["b"], df["c"])
df["agg_phi"] = df.c / df.a


def filled_circular_area_ratio(row, dims=["x", "z"]):
    """returns the area of the largest contour divided by the area of
        an encompassing circle

        useful for spheres that have reflection spots that are not captured
        by the largest contour and leave a horseshoe pattern"""
    polygons = [
        geom.MultiPoint(row.points[n][dims]).convex_hull for n in range(row.ncrystals)
    ]
    agg = shops.cascaded_union(polygons)
    area = agg.area
    poly = shops.cascaded_union(agg).convex_hull
    x, y = poly.exterior.xy
    c = cc.Cluster_Calculations(row)
    circ = c.make_circle([x[i], y[i]] for i in range(len(x)))
    circle = Point(circ[0], circ[1]).buffer(circ[2])
    x, y = circle.exterior.xy
    Ac = circle.area

    return area / Ac


df_att = df.apply(lambda x: filled_circular_area_ratio(x), axis=1)
df_att.to_hdf("df_rand_area_ratio.h5", key="area_ratio", mode="w")
