
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

class Ice_Cluster():
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


    def closest_points(self, cluster2):

        minclus2 = np.amin(cluster2.points['z'])
        maxclus1 = np.amax(self.points['z'])

        if minclus2 < maxclus1:
            diffmins = maxclus1 - minclus2
            cluster2.move([0, 0, diffmins])

        try:
            nearest_geoms_xz = nearest_points(self.projectxz(), cluster2.projectxz())
            nearest_geoms_yz = nearest_points(self.projectyz(), cluster2.projectyz())
            nearest_geoms_xy = nearest_points(self.projectxy(), cluster2.projectxy())
            #print('z from yz', nearest_geoms_yz[0].x, nearest_geoms_yz[0].y)

        except ValueError:
            return (None, None)

#         self.add_crystal(cluster2)
#         self.plot_ellipsoid_aggs([self], nearest_geoms_xz=nearest_geoms_xz, nearest_geoms_yz=nearest_geoms_yz,\
#                                  nearest_geoms_xy=nearest_geoms_xy, view='x', circle=None)
#         self.plot_ellipsoid_aggs([self], nearest_geoms_xz=nearest_geoms_xz, nearest_geoms_yz=nearest_geoms_yz,\
#                                  nearest_geoms_xy=nearest_geoms_xy, view='y', circle=None)

#         self.remove_crystal(cluster2)

        agg_yz = np.array([nearest_geoms_yz[0].x, nearest_geoms_yz[0].y]) #agg
        xtal_yz = np.array([nearest_geoms_yz[1].x, nearest_geoms_yz[1].y]) #crystal2

        stacked = np.array([self.points['y'].ravel(), self.points['z'].ravel()]).T
        tree = spatial.cKDTree(stacked)  
        distance, index = tree.query([xtal_yz], n_jobs=-1)
        agg_xyz_closest = self.points.ravel()[index]

        move_y = xtal_yz[0]-agg_yz[0]
        movez_yz = xtal_yz[1]-agg_yz[1]    

        nearpt_xz = nearest_points(Point(agg_xyz_closest['x'],agg_xyz_closest['z']), cluster2.projectxz())
#         cluster.add_crystal(cluster2)
#         cluster.plot_ellipsoid_aggs([cluster], nearest_geoms_xz=nearest_geoms_xz, nearest_geoms_yz=nearest_geoms_yz,\
#                                      nearest_geoms_xy=nearest_geoms_xy, view='x', circle=None)

#         cluster.remove_crystal(cluster2)

        move_x = nearpt_xz[1].x-nearpt_xz[0].x
        movez_xz = nearpt_xz[1].y-nearpt_xz[0].y
        cluster2.move([-move_x, -move_y, -(max(abs(movez_xz), abs(movez_yz)))])

        return (nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy)

    def _reorient(self, method='random', rotations=1):

        if method == 'IDL':
            # based on max_agg3.pro from IPAS
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                [a, b, c] = [np.random.uniform(high=np.pi / 4), np.random.uniform(high=np.pi / 4),
                             np.random.uniform(high=np.pi / 4)]
                # for mysterious reasons we are going to rotate this 3 times
                rot1 = self._euler_to_mat([a, b, c])
                rot2 = self._euler_to_mat([b * np.pi, c * np.pi, a * np.pi])
                rot3 = self._euler_to_mat([c * np.pi * 2, a * np.pi * 2, b * np.pi * 2])
                desired_rot = Quaternion(matrix=rot1 * rot2 * rot3)
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area >= max_area:
                    max_area = new_area
                    max_rot = desired_rot

                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)

        elif method == 'random':
          
            # same as schmitt but only rotating one time, with a real
            # random rotation
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                desired_rot = Quaternion.random()

                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area >= max_area:
                    max_area = new_area
                    max_rot = desired_rot

                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)
            
            self.rotation = Quaternion() # set this new rotation as the default


    def orient_cluster(self, rand_orient=False):
        # orient a cluster either randomly or to the rotation that maximizes the area
        if rand_orient:
            #self._reorient()
            xrot, yrot, zrot=random.uniform(0, 2 * np.pi),random.uniform(0, 2 * np.pi),random.uniform(0, 2 * np.pi)
            self.rotate_to([xrot, yrot, 0])   
    
        else:
#             f = lambda x: self.rotate_to([x, 0, 0], self.points).projectxy().area
#             xrot = opt.minimize_scalar(f, bounds=(0, np.pi / 2), method='Bounded', tol=1e-7).x

#             f = lambda x: self.rotate_to([0, x, 0], self.points).projectxy().area
#             yrot = opt.minimize_scalar(f, bounds=(0, np.pi / 2), method='Bounded', tol=1e-7).x
#            best_rotation = [xrot, yrot, 0]
            #print(self.points)
            
            area_og = 0
            for i in np.arange(0., np.pi/2, 0.01):
                self.rotate_to([i,0,0])
                area = self.projectxy().area
                if area > area_og:
                    xrot = i
                    area_og=area
                #self.points = self.add_points
            
            area_og = 0
            for i in np.arange(0.,np.pi/2, 0.01):
                self.rotate_to([0,i,0])
                area = self.projectxy().area
                
                if area > area_og:
                    yrot = i
                    area_og=area
                #self.points = self.add_points
            
            zrot=random.uniform(0,np.pi/2)

            best_rot = [xrot,yrot,zrot]
            self.rotate_to(best_rot)   