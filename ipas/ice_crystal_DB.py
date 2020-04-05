"""Class representing ice crystals (monomers)"""
import ipas
import copy as cp
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import shapely.geometry as geom
from pyquaternion import Quaternion
import pickle
from shapely.geometry import Point
from shapely.ops import nearest_points

class IceCrystal():

    """A hexagonal prism representing a single ice crystal.""" 
    
    def __init__(self, monomer):
        """Create an ice crystal."""
    
        self.phi = monomer.phi
        self.r = monomer.r
        self.points = np.array(pickle.loads(monomer.points), dtype=[('x', float), ('y', float), ('z', float)]) 
        self.a = (self.r*2)**3./(self.phi**(1./3.))
        self.b = 2*(((self.r*2)**3./(self.phi**(1./3.))/2.)*np.sin(60))
        self.c = self.phi*((self.r*2)**3./(self.phi**(1./3.)))
        self.center = [0, 0, 0] # start the crystal at the origin
        self.rotation = Quaternion()
        #self.move(self.center) # move the crystal
        self.tol = 10 ** -11 # used for some calculations
        self.ncrystals = 1
        self.agg_r = None
        self.agg_phi = None

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
    
    def remove_cluster(self, crystal):
        self.points = self.points[:-crystal.ncrystals]
        self.ncrystals -= crystal.ncrystals

    def _rotate_mat(self, mat):  #when a crystal is rotated, rotate the matrix with it
        points = cp.copy(self.points)
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

    def _rotate_to(self, angles):
    
        # rotate to the orientation given by the 3 angles
        # get the rotation from the current position to the desired rotation
        current_rot = self.rotation
        #print('current rotation', current_rot)
        rmat = self._euler_to_mat(angles)
        #print('rmat', rmat)
        desired_rot = Quaternion(matrix=rmat)
        #print('desired_rot', desired_rot)
        #Quarternion: a convenient mathematical notation for representing
        #orientations and rotations of objects in three dimensions
        rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
        #print('rot_mat', rot_mat)
        self._rotate_mat(rot_mat)

        # update the crystal's center:
        xyz = ['x', 'y', 'z']
        for n in range(3):
            self.center[n] = self.points[xyz[n]].mean()

        # save the new rotation
        self.rotation = desired_rot
        return self


    def _reorient(self, method='random', rotations=1):
        #reorient a crystal x random rotations to mimic IPAS in IDL instead of automatically
        #using the xrot and yrot from max area function in lap module
        #This function was only used for old runs
        #computation time is diminished using 'speedy' and bypassing this

        if method == 'IDL':
            # based on max_area2.pro from IPAS
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                [a, b, c] = [np.random.uniform(high=np.pi), np.random.uniform(high=np.pi), np.random.uniform(high=np.pi)]
                # for mysterious reasons we are going to rotate this 3 times
                rot1 = self._euler_to_mat([a, b, c])
                rot2 = self._euler_to_mat([b * np.pi, c * np.pi, a * np.pi])
                rot3 = self._euler_to_mat([c * np.pi * 2, a * np.pi * 2, b * np.pi * 2])
                desired_rot = Quaternion(matrix=rot1 * rot2 * rot3)
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area > max_area:
                    max_area = new_area
                    max_rot = desired_rot
                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)

        elif method == 'random':
            # same as IDL but only rotating one time, with a real
            # random rotation
            max_area = 0
            current_rot = self.rotation
            for i in range(rotations):
                desired_rot = Quaternion.random()
                rot_mat = (desired_rot * current_rot.inverse).rotation_matrix
                self._rotate_mat(rot_mat)
                new_area = self.projectxy().area
                if new_area > max_area:
                    max_area = new_area
                    max_rot = desired_rot
                # save our spot
                current_rot = desired_rot
            # rotate new crystal to the area-maximizing rotation
            rot_mat = (max_rot * current_rot.inverse).rotation_matrix
            self._rotate_mat(rot_mat)

        #self.rotation = Quaternion() # set this new rotation as the default
    
    def orient_crystal(self, rand_orient=False):
        #orient a crystal either randomly or to the rotation that maximizes the area
  
        if rand_orient:
            self._reorient()

        else:
            f = lambda x: -(self._rotate_to([x,0,0]).projectxy().area)
            xrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
            f = lambda x: -(self._rotate_to([0,x,0]).projectxy().area)
            yrot = opt.minimize_scalar(f, bounds=(0, np.pi/2), method='Bounded').x
            zrot=random.uniform(0, 2 * np.pi)    
            best_rot = [xrot, yrot, 0]
            self._rotate_to(best_rot)            

            
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
    
    def plot(self):
        # return a multiline object representing the edges of the prism
        lines = []
        hex1 = self.points[0:6]  #one basal face of a crystal
        hex2 = self.points[6:12]  #the other basal face

        # make the lines representing each hexagon
        for hex0 in [hex1, hex2]:
            lines.append(geom.LinearRing(list(hex0)))

        # make the lines connecting the two hexagons
        for n in range(6):
            lines.append(geom.LineString([hex1[n], hex2[n]]))

        return geom.MultiLineString(lines)
        #shapely automatically plots in jupyter notebook, no figure initialization needed

    def projectxy(self):
        return geom.MultiPoint(self.points[['x', 'y']]).convex_hull

    def projectxz(self):
        return geom.MultiPoint(self.points[['x', 'z']]).convex_hull

    def projectyz(self):
        return geom.MultiPoint(self.points[['y', 'z']]).convex_hull

    def plot_ellipsoid(self, cluster, nearest_geoms_xz=None, nearest_geoms_yz=None, nearest_geoms_xy=None, view='x', circle=None):
        
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        maxxyz = []
        minxyz = []
        for i, clus in enumerate([self, cluster]):
            
            x = np.zeros(27)
            y = np.zeros(27)
            z = np.zeros(27)
            
            if i == 0:
                color = 'r'
            else:
                color = 'k'
            X = clus.points['x']
            Y = clus.points['y']
            Z = clus.points['z']

            #for i in range(0, 360, 60):
            #    print('angle', i)

            #90, 0 for z orientation, 0, 90 for y orientation, 0, 0 for x orientation
            #ax.view_init(elev=90, azim=270)
            if view == 'x':
                ax.view_init(elev=0, azim=90)
            if view == 'y':
                ax.view_init(elev=0, azim=0)
            if view == 'z':
                ax.view_init(elev=90, azim=0)

            prismind = [0,6,7,1,2,8,9,3,4,10,11,5]  #prism lines
            i = 0
            for n in prismind:
                x[i] = X[n]
                y[i] = Y[n]
                z[i] = Z[n]
                i+=1

            ax.plot(x[0:12], y[0:12], z[0:12],color=color)

            i = 0
            for n in range(0,6): #basal face lines

                x[i+12] = X[n]
                y[i+12] = Y[n]
                z[i+12] = Z[n]
                i+=1

            x[18] = X[0]
            y[18] = Y[0]
            z[18] = Z[0]

            ax.plot(x[12:19], y[12:19], z[12:19], color=color)

            i = 0
            for n in range(6,12): #basal face lines

                x[i+19] = X[n]
                y[i+19] = Y[n]
                z[i+19] = Z[n]
                i+=1

            x[25] = X[6]
            y[25] = Y[6]
            z[25] = Z[6]

            ax.plot(x[19:26], y[19:26], z[19:26], color=color)
            
            maxX = np.max(X)
            minX = np.min(X)
            maxY = np.max(Y)
            minY = np.min(Y)
            maxZ = np.max(Z)
            minZ = np.min(Z)

            maxxyz.append(max(maxX, maxY, maxZ))
            minxyz.append(min(minX,minY,minZ))
        
        if nearest_geoms_xz != None:
            if view == 'x':
                ax.scatter(nearest_geoms_xz[0].x, nearest_geoms_yz[0].y, nearest_geoms_xz[0].y, c='red', s=100, zorder=10)
                ax.scatter(nearest_geoms_xz[1].x, nearest_geoms_yz[1].y, nearest_geoms_xz[1].y, c='k', s=100, zorder=10)
            elif view == 'y':
                ax.scatter(nearest_geoms_xz[0].x, nearest_geoms_yz[0].x, nearest_geoms_yz[0].y, c='red', s=100, zorder=10)
                ax.scatter(nearest_geoms_xz[1].x, nearest_geoms_yz[1].x, nearest_geoms_yz[1].y, c='k', s=100, zorder=10)
            else: 
                ax.scatter(nearest_geoms_xy[0].x, nearest_geoms_xy[0].y, nearest_geoms_yz[0].y, c='red', s=100, zorder=10)
                ax.scatter(nearest_geoms_xy[1].x, nearest_geoms_xy[1].y, nearest_geoms_yz[1].y, c='k', s=100, zorder=10)


        ax.set_xlim(min(minxyz), max(maxxyz))
        ax.set_ylim(min(minxyz), max(maxxyz))
        ax.set_zlim(min(minxyz), max(maxxyz))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        #ax.set_zticklabels([])
        #ax.view_init(30, i)
        #ax.view_init(0, 90)
        #plt.pause(.001)
        
        plt.show()

    def write_obj(self, filename):
        f = open(filename, 'w')
        # write the vertices
        for n in range(12):
            f.write('v ' + ' '.join(map(str, self.points[n])) + '\n')
        # write the hexagons
        for n in range(2):
            f.write('f ' + ' '.join(map(str, range(n * 6 + 1, (n + 1) * 6 + 1))) + '\n')
        for n in range(5):
            f.write('f ' + ' '.join(map(str, [n + 1, n + 2, n + 8, n + 7])) + '\n')
        f.write('f ' + ' '.join(map(str, [6, 1, 7, 12])) + '\n')
        f.close()
