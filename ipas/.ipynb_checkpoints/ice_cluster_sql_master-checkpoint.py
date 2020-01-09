
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
        
#Master Class
class Ice_Cluster():
    
    def __init__(self, ncrystals, points, n):

        # needed for bookkeeping:
        self.ncrystals = 1
        self.rotation = Quaternion()
        self.points = points

        # used for some calculations involving shapely objects
        self.tol = 10 ** -11
        # Used for the fit_ellipse function. I really do not like that
        # I have to set this so high, arrr.
        # self.tol_ellipse = 10 ** -4.5
        self.tol_ellipse = 10 ** -3
        
        self.maxz = self.points['z'].max()
        self.minz = self.points['z'].min()


    def add_cluster(self, cluster):
        self.points = np.vstack((self.points, cluster.points))
        self.ncrystals += cluster.ncrystals
        return self

    def remove_cluster(self, cluster):
        self.points = self.points[:-cluster.ncrystals]
        self.ncrystals -= cluster.ncrystals

    def move(self, xyz):
        # move the entire cluster
        self.points['x'][:self.ncrystals] += xyz[0]
        self.points['y'][:self.ncrystals] += xyz[1]
        self.points['z'][:self.ncrystals] += xyz[2]

    def max(self, dim):
        return self.points[:self.ncrystals][dim].max()

    def min(self, dim):
        return self.points[:self.ncrystals][dim].min()

    def _euler_to_mat(self, xyz):
        # Euler's rotation theorem, any rotation may be described using three angles
        [x, y, z] = xyz
        rx = np.matrix([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.matrix([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.matrix([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return rx * ry * rz

    def _rotate_mat(self, mat):
        points = cp.copy(self.points)
        self.points['x'][:self.ncrystals] = points['x'][:self.ncrystals] * mat[0, 0] + points['y'][:self.ncrystals] * \
                                            mat[0, 1] + points['z'][:self.ncrystals] * mat[0, 2]
        self.points['y'][:self.ncrystals] = points['x'][:self.ncrystals] * mat[1, 0] + points['y'][:self.ncrystals] * \
                                            mat[1, 1] + points['z'][:self.ncrystals] * mat[1, 2]
        self.points['z'][:self.ncrystals] = points['x'][:self.ncrystals] * mat[2, 0] + points['y'][:self.ncrystals] * \
                                            mat[2, 1] + points['z'][:self.ncrystals] * mat[2, 2]

    def rotate_to(self, angles):
        # rotate to the orientation given by the 3 angles
        # get the rotation from the current position to the desired
        # rotation
        current_rot = self.rotation
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
        return self  # return self to chain calls

    # def rotate_to(self, angles):
    #     # rotate the entire cluster

    #     # first get back to the original rotation
    #     if any(np.array(self.rotation) != 0):
    #         self._rev_rotate(self.rotation)

    #     # now add the new rotation
    #     self._rotate(angles)

    #     # save the new rotation
    #     self.rotation = angles

    def _center_of_mass(self):  # of cluster
        x = np.mean(self.points[:self.ncrystals]['x'])
        y = np.mean(self.points[:self.ncrystals]['y'])
        z = np.mean(self.points[:self.ncrystals]['z'])
        return [x, y, z]

    def recenter(self):
        center_move = self._center_of_mass()
        self.move([-x for x in center_move])
        return center_move

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
            # self.plot_ellipse([['x','z']])
            # cluster2.plot_ellipse([['x','z']])
            # self.plot_ellipsoid_agg_agg(cluster2)
            # print('moving cluster2 up')

            cluster2.move([0, 0, diffmins])

        try:
            nearest_geoms = nearest_points(self.projectxz(), cluster2.projectxz())
            nearest_geoms_y = nearest_points(self.projectyz(), cluster2.projectyz())

        except ValueError:
            return (None, None)

        # print('before moving')

        # self._add_cluster(cluster2)
        # self.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y, view='x')
        # self.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y, view='y')
        # self.plot_ellipsoid_agg_agg(cluster2, nearest_geoms, nearest_geoms_y, view='z')
        # self._remove_cluster(cluster2)

        movex = nearest_geoms[1].x - nearest_geoms[0].x
        movey = nearest_geoms_y[1].x - nearest_geoms_y[0].x
        movez_xz = nearest_geoms[1].y - nearest_geoms[0].y
        movez_yz = nearest_geoms_y[1].y - nearest_geoms_y[0].y
        # print(movez_xz, movez_yz)
        # print(-movex, -movey, -(max(abs(movez_xz), abs(movez_yz))))
        cluster2.move([-movex, -movey, -(max(abs(movez_xz), abs(movez_yz)))])

        return (nearest_geoms, nearest_geoms_y)

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

        return current_rot

    def orient_cluster(self, rand_orient=False):
        # orient a crystal either randomly or to the rotation that maximizes the area

        if rand_orient:
            self._reorient()

        else:

            f = lambda x: -self.rotate_to([x, 0, 0]).projectxy().area
            xrot = opt.minimize_scalar(f, bounds=(0, np.pi / 2), method='Bounded').x

            f = lambda x: -self.rotate_to([0, x, 0]).projectxy().area
            yrot = opt.minimize_scalar(f, bounds=(0, np.pi / 2), method='Bounded').x
            zrot = random.uniform(0, 2 * np.pi)

            best_rot = [xrot, yrot, zrot]

            best_rotation = [xrot, yrot, 0]
            self.rotate_to(best_rotation)
        return self

   