import glob
import sys

import agg_properties
import pandas as pd
import scripts.database as database

sys.path.append("..")


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

# creates instance of Agg property class for each row in database
# returns dictionary of attributes
out = df.apply(lambda x: agg_properties.Agg(x).get_list(), axis=1)
# convert dict to list
out1 = out.to_list()

# convert dict of attributes for all rows of database to DataFrame
df_att = pd.DataFrame(
    out1,
    columns=[
        "area_ratio",
        "convex_perim",
        "circularity",
        "roundness",
        "perim_area_ratio",
        "convexity",
        "complexity",
        "hull_area",
        "solidity",
        "equiv_d",
    ],
)

# save df of IPAS attributes
df_att.to_hdf("df_rand_attributes_include_monomers.h5", key="df_rand", mode="w")
