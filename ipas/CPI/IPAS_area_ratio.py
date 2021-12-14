"""
Calculate area ratio of IPAS aggregates from database
"""
import sys

sys.path.append("/network/rit/lab/sulialab/share/IPAS/ipas/collection_from_db")
sys.path.append("..")
sys.path.append("../collection_from_db")
import glob

import database  # noqa: E402
import pandas as pd
import shapely.geometry as geom
import shapely.ops as shops
from dask import dataframe as dd
from shapely.geometry import Point

import ipas.cluster_calculations as cc  # noqa: E402


def filled_circular_area_ratio(row, dims=["x", "z"]):
    """returns the area of the largest contour divided by the area of
    an encompassing circle

    useful for spheres that have reflection spots that are not captured
    by the largest contour and leave a horseshoe pattern (for CPI data)"""

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

# slow!
ddf = dd.from_pandas(df, npartitions=8)
df_ar = df.apply(lambda x: filled_circular_area_ratio(x), axis=1)

# save area ratio dataframe so we don't have to rerun
df_ar.to_hdf("df_rand_only_area_ratio.h5", key="area_ratio", mode="w")

# read back in
df_ar = pd.read_hdf("df_rand_only_area_ratio.h5").reset_index(drop=True)
# convert h5 to pandas and rename 0 column name to area ratio
df_ar = pd.DataFrame(df_ar).rename(columns={0: "area_ratio"})

# concatenate area ratio with IPAS dataframe (but without points to save time reading in)
df = df.drop(columns="points").reset_index(drop=True)
dfc = pd.concat([df, df_ar], axis=1)

# save df of IPAS attributes with area ratio
dfc.to_hdf("df_IPAS_rand_area_ratio_no_points.h5", key="df_IPAS_att", mode="w")
