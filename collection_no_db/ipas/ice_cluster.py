
"""Class representing ice clusters or aggregates. Parent class that defines point arrays, 
crystals within the cluster, and methods to move and reorient the arrays"""

import ipas
import numpy.linalg as la
import math
import shapely.ops as shops
from pyquaternion import Quaternion
import copy as cp
import numpy as np
import scipy.optimize as opt
import shapely.geometry as geom
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
from descartes.patch import PolygonPatch
import descartes
import time
from shapely.ops import nearest_points
from matplotlib.patches import Ellipse
import operator
from scipy import spatial         
import ipas

class IceCluster():
    
    def __init__(self, crystal, size=1):

        self.ncrystals = 1
        self.rotation = Quaternion()
        self.points = np.full((1, 12), np.nan,
                              dtype=[('x', float), ('y', float), ('z', float)])
        self.points[0] = crystal.points
#         self.xs = None
#         self.ys = None
#         self.zs = None

        self.add_points = None
        self.orient_points = None
        self.mono_phi = crystal.phi
        self.mono_r = crystal.r
        self.tol_ellipse = 10 ** -3
        

    def to_dict(self):
        return {
             'ncrystals':self.ncrystals,
#              'xs':self.xs,
#              'ys':self.ys,
#              'zs':self.zs,
             'points': self.points,
             'a':self.agg_a,
             'b':self.agg_b,
             'c':self.agg_c, 
             'cplx':self.cplx,
             'phi2D':self.phi2D,
             'mono_phi':self.mono_phi,
             'mono_r':self.mono_r
        }

    def add_cluster(self, cluster):
        self.points = np.vstack((self.points, cluster.points))
        self.ncrystals += cluster.ncrystals
        return self


    def remove_cluster(self, cluster):
        self.points = self.points[:-cluster.ncrystals]
        self.ncrystals -= cluster.ncrystals
        

    def add_crystal(self, crystal):
        self.points = np.vstack((self.points, crystal.points))
        self.ncrystals += crystal.ncrystals
        return self  #to make clus 3 instance
    

    def remove_crystal(self, crystal):
        self.points = self.points[:-crystal.ncrystals]
        self.ncrystals -= crystal.ncrystals


    def move(self, xyz):
        # move the entire cluster 
        self.points['x'] += xyz[0]
        self.points['y'] += xyz[1]
        self.points['z'] += xyz[2]


    def _euler_to_mat(self, xyz):
        # Euler's rotation theorem, any rotation may be described using three angles
        [x, y, z] = xyz
        rx = np.matrix([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.matrix([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.matrix([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return rx * ry * rz


    def _rotate_mat(self, mat):
        points = cp.copy(self.points)
        self.points['x'] = points['x'] * mat[0, 0] + points['y'] * mat[0, 1] + points['z'] * mat[0, 2]
        self.points['y'] = points['x'] * mat[1, 0] + points['y'] * mat[1, 1] + points['z'] * mat[1, 2]
        self.points['z'] = points['x'] * mat[2, 0] + points['y'] * mat[2, 1] + points['z'] * mat[2, 2]


    def rotate_to(self, angles):
        # rotate to the orientation given by the 3 angles
        # get the rotation from the current position to the desired
        # rotation
        current_rot = self.rotation
        #print('current_rot', current_rot)
        rmat = self._euler_to_mat(angles)
        desired_rot = Quaternion(matrix=rmat)
        # print('desired', desired_rot)
        # euler_angles = self.quaternion_to_euler(desired_rot[0], desired_rot[1], desired_rot[2], desired_rot[3])
        # print('q to euler', euler_angles)
        rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
        self._rotate_mat(rot_mat)

        # save the new rotation
        self.rotation = desired_rot
        # self.saverot = euler_angles

    # def rotate_to(self, angles):
    #     # rotate the entire cluster

    #     # first get back to the original rotation
    #     if any(np.array(self.rotation) != 0):
    #         self._rev_rotate(self.rotation)

    #     # now add the new rotation
    #     self._rotate(angles)

    #     # save the new rotation
    #     self.rotation = angles


    def center_of_mass(self):  # of cluster
        x = np.mean(self.points[:self.ncrystals]['x'])
        y = np.mean(self.points[:self.ncrystals]['y'])
        z = np.mean(self.points[:self.ncrystals]['z'])
        return [x, y, z]
    

    def recenter(self):
        center_move = self.center_of_mass()
        self.move([-x for x in center_move])
        return center_move
    

    def _crystal_projectxy(self, n):
        return geom.MultiPoint(self.points[n][['x', 'y']])


    def _crystal_projectxz(self, n):
        return geom.MultiPoint(self.points[n][['x', 'z']])


    def _crystal_projectyz(self, n):
        return geom.MultiPoint(self.points[n][['y', 'z']])


    def projectxy(self):
        polygons = [self._crystal_projectxy(n) for n in range(self.ncrystals)]
        return shops.cascaded_union(polygons)


    def projectxz(self):
        polygons = [self._crystal_projectxz(n) for n in range(self.ncrystals)]
        return shops.cascaded_union(polygons)


    def projectyz(self):
        polygons = [self._crystal_projectyz(n) for n in range(self.ncrystals)]
        return shops.cascaded_union(polygons)


    def generate_random_point_fast(self, new_crystal, number=1):

        # print(self.ncrystals)
        crystals = [self, new_crystal]
        list_of_points = []
        for i in crystals:
            # NEW CRYSTAL RANDOM PT
            polygon = i.projectxy()
            minx, miny, maxx, maxy = polygon.bounds

            # choose random point over polygon
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            # point doesn't need to hit polygon for agg-agg since moving to closest anyways
            list_of_points.append(pnt)

        agg_pt = list_of_points[0]
        new_pt = list_of_points[1]
        
        return (agg_pt, new_pt)


    def find_x_nearest_geoms(self, cluster2, yz_pt):
        '''
        find x point that corresponds to (y,z) nearest pt on particle
        since shapely only deals in 2 dimensions and IPAS polygons 
        are only made up of vertices, we need to find all 3 dimensions
        of the closest points between particles assuming the closest 
        points do not land on vertices
        '''
        
        if yz_pt[0] in self.points['y']:  # closest point is on vertex
            A = self.points[self.points['y']==yz_pt[0]]
            return (A['x'], A['y'], A['z'])
            
        else:  # closest point is on edge
            stacked = np.array([self.points['y'].ravel(),
                            self.points['z'].ravel()]).T
            # uses cKDTree (written in C++, faster than KDTree):
            # provides an index into a set of k-dimensional points
            # which can be used to rapidly look up the nearest neighbors of any point
            stacked = np.array([self.points['y'].ravel(),
                                self.points['z'].ravel()]).T

            tree = spatial.cKDTree(stacked)
            # grab k of the closest vertices to check if the yz_pt
            # lands on a line between 2 vertices
            distance, index = tree.query([yz_pt], k=12, n_jobs=-1)
            
            A = self.points.ravel()[index[0,0]] # vertex A (closest)
            
            i = 1
            is_btwn = False
            while is_btwn is False:
                B = self.points.ravel()[index[0,i]]
                # point yz_pt lies between two vertices along an edge if:
                # the area of a triangle between the two vertices and point is 0
                # A and B are vertices
                area = np.abs((A['y']*(B['z']-yz_pt[1])+
                               B['y']*(yz_pt[1]-A['z'])+
                               yz_pt[0]*(A['z']-B['z']))/2.00)
                
                if area < 5.0:
                    is_btwn = True
                    # subtract B and A to get vector direction
                    AB = (B['x']-A['x'], B['y']-A['y'], B['z']-A['z'])
                    slope = (yz_pt[0] - A['y'])/AB[1]

                    # find x using slope and vector direction
                    x = A['x'] + slope*AB[0]

                    xyz = (x, yz_pt[0], yz_pt[1])
                    return xyz
                else:
                    i+=1
                    
    def crystals(self, i=None):
        # return a crystal with the same points and attributes as the
        # nth crystal in the cluster
        if i is None:
            crystals = []
            for n in range(self.ncrystals):
                cr = ipas.IceCrystal(1, 1)
                cr.points = self.points[n]
                cr.rotation = self.rotation
                cx = cr.points['x'].mean()
                cy = cr.points['y'].mean()
                cz = cr.points['z'].mean()
                cr.center = [cx, cy, cz]
                cr.maxz = cr.points['z'].max()
                cr.minz = cr.points['z'].min()
                crystals.append(cr)
            return crystals
        else:
            cr = ipas.IceCrystal(1, 1)
            cr.points = self.points[i]  #i = 0
            cr.rotation = self.rotation
            cx = cr.points['x'].mean()
            cy = cr.points['y'].mean()
            cz = cr.points['z'].mean()
            cr.center = [cx, cy, cz]
            cr.maxz = cr.points['z'].max()
            cr.minz = cr.points['z'].min()
            return cr


    def closest_points_old(self, crystal, lodge=0):
        # drop a new crystal onto the cluster
        # use the bounding box to determine which crystals to get
        xmax = max(crystal.projectxy().exterior.coords.xy[0])
        ymax = max(crystal.projectxy().exterior.coords.xy[1])
        xmin = min(crystal.projectxy().exterior.coords.xy[0])
        ymin = min(crystal.projectxy().exterior.coords.xy[1])


        close = np.all([self.points['x'][:self.ncrystals].max(axis=1) >= xmin,
                        self.points['x'][:self.ncrystals].min(axis=1) <= xmax,
                        self.points['y'][:self.ncrystals].max(axis=1) >= ymin,
                        self.points['y'][:self.ncrystals].min(axis=1) <= ymax], axis=0)


        which_close = np.where(close)

        close_crystals = [ self.crystals(n) for n in which_close[0] ]

        # see which crystals could actually intersect with the new crystal
        close_crystals = [ x for x in close_crystals if x.projectxy().intersects(crystal.projectxy()) ]

        # close_crystals = [ x for x in self.crystals() if x.projectxy().intersects(newpoly) ]
        if len(close_crystals) == 0:
            return False # the crystal missed!

        # we know highest hit is >= max(minzs), therefore the first
        # hit can't be below (max(minzs) - height(crystal))
        minzs = [ crystal2.minz for crystal2 in close_crystals ]
        first_hit_lower_bound = max(minzs) - (crystal.maxz - crystal.minz)
        # remove the low crystals, sort from highest to lowest
        close_crystals = [ x for x in close_crystals if x.maxz > first_hit_lower_bound ]
        close_crystals.sort(key=lambda x: x.maxz, reverse=True)

        # look to see where the new crystal hits the old ones
        mindiffz = crystal.minz - first_hit_lower_bound # the largest it can possibly be

        for crystal2 in close_crystals:
            if first_hit_lower_bound > crystal2.maxz:
                break # stop looping if the rest of the crystals are too low
            diffz = crystal.min_vert_dist(crystal2)
            if diffz is None:
                break

            #return diffz
            # update if needed
            if diffz < mindiffz:
                mindiffz = diffz
                first_hit_lower_bound = crystal.minz - mindiffz
                # take the highest hit, move the crystal to that level
        crystal.move([0, 0, -mindiffz])

        # append new crystal to list of crystals
        # self.crystals.append(crystal)
        self.add_crystal(crystal)
        # fin.
        return True

    def closest_points(self, cluster2, plot_closest_pts=False):
        '''
        aggregate two particles using the nearest points
        Shapely function
        '''
        
        # make sure cluster 2 is above cluster 1
        minclus2 = np.amin(cluster2.points['z'])
        maxclus1 = np.amax(self.points['z'])

        # this makes columns always collect basal to prism face
        if (minclus2 < maxclus1) and (cluster2.phi < 1.0):
            diffmins = maxclus1 - minclus2
            cluster2.move([0, 0, diffmins])

        # find nearest points in xz, yz, and xy
        try:
            nearest_geoms_xz = nearest_points(self.projectxz(),
                                              cluster2.projectxz())
            nearest_geoms_yz = nearest_points(self.projectyz(),
                                              cluster2.projectyz())
            nearest_geoms_xy = nearest_points(self.projectxy(),
                                              cluster2.projectxy())
            #print('z from yz', nearest_geoms_yz[0].x,
            #      nearest_geoms_yz[0].y)

        except ValueError:
            return (None, None)
        
        # -------------------
        if plot_closest_pts:
            print('BEFORE MOVING')
            self.plot_closest_pts(cluster2, 
                                  nearest_geoms_xz,
                                  nearest_geoms_yz,
                                  nearest_geoms_xy)
        
        # first get closest points on cluster and monomer in y-z
        # NOT NECESSARILY ON VERTEX
        cluster_yz = np.array([nearest_geoms_yz[0].x,
                           nearest_geoms_yz[0].y])
        monomer_yz = np.array([nearest_geoms_yz[1].x,
                            nearest_geoms_yz[1].y]) 

        # find the difference between points to move in y and z
        move_y = monomer_yz[0] - cluster_yz[0]
        movez_yz = monomer_yz[1] - cluster_yz[1]
        
        # get (x,y,z) pt on monomer closest to agg
        xyz_mono = cluster2.find_x_nearest_geoms(yz_pt=monomer_yz)
        # get (x,y,z) pt on aggregate closest to monomer
        #xyz_agg = self.find_x_nearest_geoms(cluster2, yz_pt=cluster_yz)
        xyz_agg = (0,0,0)
        
        # now that the x pt is found on both particles, 
        # find the distance needed to move cluster2 in x to meet agg
        move_x = xyz_mono[0]-xyz_agg[0]

        cluster2.move([-move_x,-move_y, -movez_yz])
        
        # -------------------
        if plot_closest_pts:
            print('AFTER MOVING')
            self.plot_closest_pts(cluster2, 
                                  nearest_geoms_xz,
                                  nearest_geoms_yz,
                                  nearest_geoms_xy)
        
        return (nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy)
    

    def plot_closest_pts(self, cluster2, nearest_geoms_xz, nearest_geoms_yz,
                        nearest_geoms_xy):
                
        self.add_crystal(cluster2)
        self.plot_ellipsoid_aggs([self], nearest_geoms_xz,
                                 nearest_geoms_yz,
                                 nearest_geoms_xy,
                                 view='x')
        self.plot_ellipsoid_aggs([self], nearest_geoms_xz,
                                 nearest_geoms_yz,
                                 nearest_geoms_xy,
                                 view='y')

        self.remove_crystal(cluster2)


    def orient_cluster(self, rand_orient=False):
        '''
        orient a cluster either randomly
        or to the rotation that maximizes
        the area in the xy plane
        '''
        if rand_orient:  # random
            xrot = random.uniform(0, 2 * np.pi)
            yrot = random.uniform(0, 2 * np.pi)
            zrot = random.uniform(0, 2 * np.pi)
            self.rotate_to([xrot, yrot, 0])   
    
        else:  # flat
            area_og = 0
            for i in np.arange(0., np.pi/2, 0.01):
                self.rotate_to([i,0,0])
                area = self.projectxy().area
                if area > area_og:
                    xrot = i
                    area_og=area
                self.points = self.add_points
            
            area_og = 0
            for i in np.arange(0.,np.pi/2, 0.01):
                self.rotate_to([0,i,0])
                area = self.projectxy().area
                
                if area > area_og:
                    yrot = i
                    area_og=area
                self.points = self.add_points
            
            zrot=random.uniform(0,np.pi/2)

            best_rot = [xrot,yrot,zrot]
            self.rotate_to(best_rot)   