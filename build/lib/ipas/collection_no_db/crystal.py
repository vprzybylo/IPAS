"""
Class representing ice crystals (monomers)
"""

import copy as cp
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import shapely.geometry as geom
from pyquaternion import Quaternion
from shapely.geometry import Point
from shapely.ops import nearest_points
import numpy.linalg as la
from scipy import spatial


class Crystal():
    """A hexagonal prism representing a single ice crystal.""" 

    def __init__(self, a, c , center=[0, 0, 0], rotation=[0, 0, 0]):
        self.a = a
        self.c = c
        self.phi = self.c/self.a
        self.r = int(np.round(np.power((np.power(self.a,2)*self.c),(1./3.))))
        self.center = [0, 0, 0] # start the crystal at the origin
        self.rotation = Quaternion()
        self.ncrystals = 1
        self.hold_clus = None
        self.tol = 10 ** -11 # used for some calculations

        # put together the hexagonal prism
        ca = c*2  #diameter
        mf = a*2  #diameter
        f = np.sqrt(3) / 4 # convenient number for hexagons
        x1 = ca / 2

        #creates 12 point arrays for hexagonal prisms

        if c < a:  #initialize plates so that the basal face is falling down
            self.points = np.array([(mf*f, -mf / 4, x1), (mf * f, mf / 4, x1),
                        (0, mf / 2, x1), (-mf * f, mf / 4, x1),
                        (-mf * f, -mf / 4, x1), (0, -mf/2, x1),
                        (mf * f, -mf / 4, -x1), (mf * f, mf / 4, -x1),
                        (0, mf / 2, -x1), (-mf * f, mf / 4, -x1),
                        (-mf * f, -mf / 4, -x1), (0, -mf/2, -x1)],
                       dtype=[('x', float), ('y', float), ('z', float)])


        else:  #initialize points so that columns fall prism face down
            self.points = np.array([(x1, -mf / 4, mf * f), (x1, mf / 4, mf * f),
                        (x1, mf / 2, 0), (x1, mf / 4, -mf * f),
                        (x1, -mf / 4, -mf * f), (x1, -mf/2, 0),
                        (-x1, -mf / 4, mf * f), (-x1, mf / 4, mf * f),
                        (-x1, mf / 2, 0), (-x1, mf / 4, -mf * f),
                        (-x1, -mf / 4, -mf * f), (-x1, -mf/2, 0)],
                       dtype=[('x', float), ('y', float), ('z', float)])

        self.rotate_to(rotation) # rotate the crystal
        self.move(center) # move the crystal to center
        self.maxz = self.points['z'].max()
        self.minz = self.points['z'].min()
        

    def move(self, xyz):  #moves the falling crystal anywhere over the seed crystal/aggregate within the max bounds
        self.points['x'] += xyz[0]
        self.points['y'] += xyz[1]
        self.points['z'] += xyz[2]
        # update the crystal's center:
        for n in range(3):
            self.center[n] += xyz[n]


    def _center_of_mass(self):
        x = np.mean(self.points['x'])
        y = np.mean(self.points['y'])
        z = np.mean(self.points['z'])
        return [x, y, z]


    def recenter(self):
        self.move([ -x for x in self._center_of_mass() ])
    

    def add_crystal(self, crystal):
        self.points = np.vstack((self.points, crystal.points))
        self.ncrystals += crystal.ncrystals
        return self  #to make clus 3 instance
    

    def remove_crystal(self, crystal):
        self.points = self.points[:-crystal.ncrystals]
        self.ncrystals -= crystal.ncrystals
    

    def remove_cluster(self, crystal):
        self.points = self.points[:-crystal.ncrystals]
        self.ncrystals -= crystal.ncrystals


    def _rotate_mat(self, mat):  #when a crystal is rotated, rotate the matrix with it
        points=cp.deepcopy(self.points)
        self.points['x'] = points['x'] * mat[0, 0] + points['y'] * mat[0, 1] + points['z'] * mat[0, 2]
        self.points['y'] = points['x'] * mat[1, 0] + points['y'] * mat[1, 1] + points['z'] * mat[1, 2]
        self.points['z'] = points['x'] * mat[2, 0] + points['y'] * mat[2, 1] + points['z'] * mat[2, 2]
        

    def _euler_to_mat(self, xyz):
        #Euler's rotation theorem, any rotation may be described using three angles.
        #takes angles and rotates coordinate system
        [x, y, z] = xyz
        rx = np.matrix([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.matrix([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.matrix([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return rx * ry * rz


    def rotate_to(self, angles):
    
        # rotate to the orientation given by the 3 angles
        # get the rotation from the current position to the desired rotation
        
        rmat = self._euler_to_mat(angles)
        desired_rot = Quaternion(matrix=rmat)
        
        rot_mat = (desired_rot * self.rotation.inverse).rotation_matrix
        self._rotate_mat(rot_mat)

        # update the crystal's center:
        xyz = ['x', 'y', 'z']
        for n in range(3):
            self.center[n] = self.points[xyz[n]].mean()
        self.rotation = desired_rot
        return self


    def orient_crystal(self, rand_orient=False):
        '''
        orient a crystal either randomly or to the
        rotation that maximizes the area in the x-y plane
        '''

        if rand_orient:
            xrot = random.uniform(0, 2 * np.pi)
            yrot = random.uniform(0, 2 * np.pi)
            zrot = random.uniform(0, 2 * np.pi)
            self.rotate_to([xrot, yrot, zrot])

        else:

            area_og = 0
            for i in np.arange(0.,np.pi/2, 0.1):
                self.rotate_to([i,0,0])
                area = self.projectxy().area
                if area > area_og:
                    xrot = i
                    area_og=area
                self.points = self.hold_clus

            area_og = 0
            for i in np.arange(0.,np.pi/2, 0.1):
                self.rotate_to([0,i,0])
                area = self.projectxy().area
                if area > area_og:
                    yrot = i
                    area_og=area
                self.points = self.hold_clus
            
            zrot=random.uniform(0, 2 * np.pi)
            best_rot = [xrot,yrot,zrot]
            self.rotate_to(best_rot)

            
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
            #self.plot_ellipsoid()
            #cluster2.plot_ellipsoid()
            #print('moving cluster2 up')
            cluster2.move([0, 0, diffmins])

        try:
            clus1projxz = self.projectxz()
            clus2projxz = cluster2.projectxz()
            clus1projyz = self.projectyz()
            clus2projyz = cluster2.projectyz()
            clus1projxy = self.projectxy()
            clus2projxy = cluster2.projectxy()

            nearest_geoms_xz = nearest_points(clus1projxz, clus2projxz)
            nearest_geoms_yz = nearest_points(clus1projyz, clus2projyz)
            nearest_geoms_xy = nearest_points(clus1projxy, clus2projxy)
        except ValueError:
            return (None, None)  
                
        movex = nearest_geoms_xz[1].x - nearest_geoms_xz[0].x
        movey = nearest_geoms_yz[1].x - nearest_geoms_yz[0].x
        movez_xz = nearest_geoms_xz[1].y - nearest_geoms_xz[0].y
        movez_yz = nearest_geoms_yz[1].y - nearest_geoms_yz[0].y
        
        #print('movez_xz, movez_yz', movez_xz, movez_yz)
        cluster2.move([-movex, -movey, -(max(abs(movez_xz), abs(movez_yz)))])
        #print('pts1', cluster2.points['x'].max(), cluster2.points['y'].max())
        #move in x-y
        movex = nearest_geoms_xy[1].x - nearest_geoms_xy[0].x
        movey = nearest_geoms_xy[1].y - nearest_geoms_xy[0].y
        #if movex != 0.0 or movey != 0.0:
        #    print('moving x-y', movex, movey)
            
        cluster2.move([-movex, -movey, 0])

        return (nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy)
    
    
    def evenly_spaced_mesh(self, nx=10):

        xi = np.linspace(np.min(self.points['x']),
                         np.max(self.points['x']),nx)
        yi = np.linspace(np.min(self.points['y']),
                         np.max(self.points['y']),nx)
        zi = np.linspace(np.min(self.points['z']),
                         np.max(self.points['z']),nx)

        x, y, z = np.meshgrid(xi, yi, zi)
        return x, y, z
    

    def closest_point_mesh(self, shapely_pt, x, y, z):
        '''
        find (x,y,z) on mesh grid that is closest to
        the closest points on the monomer returned from shapely
        '''
        # use mesh as target points to search
        # transpose such that the array isnt all y then all z
        # but instead pairs of (y,z)
        grid_pts = np.array([y.ravel(),z.ravel()]).T
        tree = spatial.cKDTree(grid_pts)
        # grab k of the closest vertices to point returned from shapely
        # can be on any edge, vertex, or surface/face
        distance, index = tree.query([shapely_pt], k=1, n_jobs=-1)
        # now we can use the index to find x,y,z pt on mesh
        x_closest = x.ravel()[index[0]]
        y_closest = y.ravel()[index[0]]
        z_closest = z.ravel()[index[0]]

        return (x_closest, y_closest, z_closest)
    

    def projectxy(self):
        return geom.MultiPoint(self.points[['x', 'y']]).convex_hull


    def projectxz(self):
        return geom.MultiPoint(self.points[['x', 'z']]).convex_hull


    def projectyz(self):
        return geom.MultiPoint(self.points[['y', 'z']]).convex_hull
    

    def _mvee(self, tol=0.01):  # mve = minimum volume ellipse
        # Based on work by Nima Moshtagh
        # http://www.mathworks.com/matlabcentral/fileexchange/9542

        """
        Finds the ellipse equation in "center form"
        (x-c).T * A * (x-c) = 1
        """

        # print('points_Arr', points_arr)
        points_arr = np.array([list(i) for i in self.points])
        N, d = points_arr.shape
        Q = np.column_stack((points_arr, np.ones(N))).T

        err = tol + 1.0
        u = np.ones(N) / N
        while err > tol:
            # assert u.sum() == 1 # invariant
            X = np.dot(np.dot(Q, np.diag(u)), Q.T)
            M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u

        c = np.dot(u, points_arr)

        A = la.inv(np.dot(np.dot(points_arr.T, np.diag(u)), points_arr)
                   - np.multiply.outer(c, c)) / d
        return A, c


    def ellipsoid_axes(self):
        '''
        semi-principal axes of fit-ellipsoid 
        in descending order
        '''
        A, c = self._mvee()
        U, D, V = la.svd(A)  # singular-value decomposition
        rx, ry, rz = 1. / np.sqrt(D)  # D is a diagonal matrix
        self.agg_a, self.agg_b, self.agg_c = sorted([rx, ry, rz], reverse=True)
        return self.agg_a, self.agg_b, self.agg_c
    
        
    def plot_crystal(self, ax, color):  
        #plots individual monomers
        
        x = np.zeros(27)
        y = np.zeros(27)
        z = np.zeros(27)

        X = self.points['x']
        Y = self.points['y']
        Z = self.points['z']

        
        prismind = [0, 6, 7, 1, 2, 8, 9, 3, 4, 10, 11, 5]  # prism lines
        i = 0
        for n in prismind:
            x[i] = X[n]
            y[i] = Y[n]
            z[i] = Z[n]
            i += 1
  
        ax.plot(x[0:12], y[0:12], z[0:12], color=color)

        i = 0
        for n in range(0, 6):  # basal face lines

            x[i + 12] = X[n]
            y[i + 12] = Y[n]
            z[i + 12] = Z[n]
            i += 1

        x[18] = X[0]
        y[18] = Y[0]
        z[18] = Z[0]
        
        ax.plot(x[12:19], y[12:19], z[12:19], color=color)

        i = 0
        for n in range(6, 12):  # basal face lines

            x[i + 19] = X[n]
            y[i + 19] = Y[n]
            z[i + 19] = Z[n]
            i += 1

        x[25] = X[6]
        y[25] = Y[6]
        z[25] = Z[6]

        ax.plot(x[19:26], y[19:26], z[19:26], color=color)