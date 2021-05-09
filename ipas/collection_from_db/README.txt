5/7/21

Collection_from_db provides functionality for ice-aggregate
and aggregate-aggregate collection from the pre-saved db
of aggregates (database files stored in ./db_files)

Modules:
-------
batch_statistics:
    - Calculates statistics on a group of IPAS particles:
        - mean +/- one std
        - min and max of batch
        - mode of histogram
        - characteristic value of gamma distribution
        
calculations:
    - Finds fit ellipse and ellipsoids surrounding clusters
    - Performs any external calculations on the clusters 
        - For example, aspect ratio, complexity, etc.
        
database:
    - Read database of IPAS aggregates
        - with and without point arrays
    - Get average stats on the entire database

cluster:
    - Class representing ice clusters or aggregates
        - multiple monomers
    - Defines point arrays, crystals within the cluster, and methods to move and reorient the arrays

crystal:
    - Class representing ice crystals (monomers)
    - Initializes hexagonal prisms
    - Performs orientation rotations
    - Gets projections in all planes

aggagg_collection:
    - Runs aggregate-aggregate collection
        - calls all class methods in the proper order
        - returns relevant data after collection

iceagg_collection:
    - Runs ice-aggregate collection
        - calls all class methods in the proper order
        - returns relevant data after collection
        
plot:
    - Sub class to Cluster
        - holds methods to plot the aggregate(s)
        - no interactive plots
        - identical to plot.py in collection_no_db

plotly_aggs:
    - plots aggregates from database using plotly
    - creates interactive plots and provides a URL for html rendering