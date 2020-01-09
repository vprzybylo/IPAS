
"""Class representing ice crystals (monomers)"""
from sqlalchemy import Column, Float, Integer, Boolean
from sqlalchemy.orm import relationship
import ipas
import numpy.linalg as la
import math
import shapely.ops as shops
from pyquaternion import Quaternion
import copy as cp
import numpy as np
import scipy.optimize as opt
import shapely.geometry as geom
import shapely.affinity as sha
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
from descartes.patch import PolygonPatch
import descartes
import time
from shapely.ops import nearest_points
from sqlalchemy import Column, Float, Integer, ForeignKey, PickleType
from base import Base
from matplotlib.patches import Ellipse
import operator
        
#Child
class IceCluster(Base):
    __tablename__ = 'aggregates'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ncrystals = Column(Integer)
    points = Column(PickleType)
    agg_phi = Column(Float)
    agg_r = Column(Float)
    a = Column(Float)
    b = Column(Float)
    c = Column(Float)
    cplx = Column(Float)
    phi2D = Column(Float)
    crystal = relationship('IceCrystal', back_populates='aggs') #works
    #crystal_id = Column(Integer, ForeignKey('crystals.id'))

    def __init__(self, crystal, n, size=1):

        # needed for bookkeeping:
        self.ncrystals = 1
        self.rotation = Quaternion()
        self.points = np.full((size, 12), np.nan,
                              dtype=[('x', float), ('y', float), ('z', float)])
        self.points[0] = crystal.points
        self.size = size
        # used for some calculations involving shapely objects
        self.tol = 10 ** -11
        # Used for the fit_ellipse function. I really do not like that
        # I have to set this so high, arrr.
        # self.tol_ellipse = 10 ** -4.5
        self.tol_ellipse = 10 ** -3
        self.major_axis = {}
        self.minor_axis = {}
        self.maxz = self.points['z'].max()
        self.minz = self.points['z'].min()

    def _crystals(self, i=None):
        # return a crystal with the same points and attributes as the
        # nth crystal in the cluster
        if i is None:
            crystals = []
            for n in range(self.ncrystals):
                cr = IceCrystal(1, 1)
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
            cr = IceCrystal(1, 1)
            cr.points = self.points[i]  # i = 0
            cr.rotation = self.rotation
            cx = cr.points['x'].mean()
            cy = cr.points['y'].mean()
            cz = cr.points['z'].mean()
            cr.center = [cx, cy, cz]
            cr.maxz = cr.points['z'].max()
            cr.minz = cr.points['z'].min()
            return cr

    def add_crystal(self, crystal):
        n = self.ncrystals
        if n < self.size:
            self.points[n] = crystal.points
        else:
            # self.points = np.append(self.points, [crystal.points], axis=0)
            self.points = np.vstack((self.points, crystal.points))
        self.ncrystals += 1

        return self

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

    def plot(self):
        return geom.MultiLineString([lines for crystal in self.crystals() for lines in crystal.plot()])

    def _crystal_projectxy(self, n):
        return geom.MultiPoint(self.points[n][['x', 'y']]).convex_hull

    def _crystal_projectxz(self, n):
        return geom.MultiPoint(self.points[n][['x', 'z']]).convex_hull

    def _crystal_projectyz(self, n):
        return geom.MultiPoint(self.points[n][['y', 'z']]).convex_hull

    def projectxy(self):
        polygons = [self._crystal_projectxy(n) for n in range(self.ncrystals)]
        return shops.cascaded_union(polygons)

    def projectxz(self):
        polygons = [self._crystal_projectxz(n) for n in range(self.ncrystals)]
        return shops.cascaded_union(polygons)

    def projectyz(self):
        polygons = [self._crystal_projectyz(n) for n in range(self.ncrystals)]
        return shops.cascaded_union(polygons)

    def _max_xtal_dist_ellipse(self, crystal):
        paramsagg = self.fit_ellipse([['x', 'y']])
        paramsnew = self.fit_ellipse([['x', 'y']])
        maxdimagg = max([paramsagg['width'], paramsagg['height']])
        maxdimnew = max([paramsnew['width'], paramsnew['height']])
        return (maxdimagg, maxdimnew)

    def max_xtal_dist(self, crystal):
        crystals1 = [self, crystal]
        dmaxclus = []
        dmaxnew = []
        n = 0
        for i in crystals1:

            x, y = list(i.projectxy().exterior.coords.xy)

            dinit = 0
            for j in range(len(x)):
                for l in range(len(x)):
                    d = (Point(x[l], y[l]).distance(Point(x[j], y[j])))
                    if d > dinit:
                        dinit = d

                        if n == 0:
                            dmaxclus.append(d)
                        if n == 1:
                            dmaxnew.append(d)
                        xstart = l
                        ystart = l
                        xend = j
                        yend = j

            if n == 0:
                dmaxclus = max(dmaxclus)

            if n == 1:
                dmaxnew = max(dmaxnew)

            n += 1
        return (dmaxclus, dmaxnew)

    def generate_random_point(self, new_crystal, number=1):

        # print(self.ncrystals)
        crystals = [self, new_crystal]
        list_of_points = []
        for i in crystals:
            # NEW CRYSTAL RANDOM PT
            polygon = i.projectxy()
            minx, miny, maxx, maxy = polygon.bounds
            hit = 0
            miss = 0
            while hit < number:
                # choose random point over polygon
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(pnt):
                    list_of_points.append(pnt)
                    hit += 1
                else:
                    # print('agg missed in poly', miss)
                    miss += 1

        agg_pt = list_of_points[0]
        new_pt = list_of_points[1]
        return (agg_pt, new_pt)

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

    def calculate_S_ratio_fast(self, plates, crystal):
        # Calculate separation of crystals for further collection restriction
        start = time.clock()
        crystals1 = [self, crystal]  # self is the monomer to attach
        # crystal is the preexisting monomer or agg
        d = []
        for i in crystals1:
            x, y = list(i.projectxy().exterior.coords.xy)
            maxx = max(x)
            minx = min(x)
            maxy = max(y)
            miny = min(y)

            d.append(Point(maxx, maxy).distance(Point(minx, miny)))

        l = (self.projectxy().centroid).distance(crystal.projectxy().centroid)
        S = 2 * l / (d[0] + d[1])

        if plates:
            lmax = 0.6 * (d[0] + d[1]) / 2  # S parameter can't be higher than 0.6 for plates
        else:
            lmax = 0.3 * (d[0] + d[1]) / 2  # S parameter can't be higher than 0.3 for columns

        end = time.clock()
        # print("fast %2f"%(end-start))
        return S, lmax

    def _extract_poly_coords(self, geom):
        if geom.type == 'Polygon':
            exterior_coords = geom.exterior.coords[:]
            interior_coords = []
            for interior in geom.interiors:
                interior_coords += interior.coords[:]
        elif geom.type == 'MultiPolygon':
            exterior_coords = []
            interior_coords = []
            for part in geom:
                epc = self.extract_poly_coords(part)  # Recursive call
                exterior_coords += epc['exterior_coords']
                interior_coords += epc['interior_coords']
        else:
            raise ValueError('Unhandled geometry type: ' + repr(geom.type))
        return {'exterior_coords': exterior_coords,
                'interior_coords': interior_coords}

    def closest_points(self, cluster2):

        # clus1_ext = self.projectxz().exterior
        # maxzclus1 = max(clus1_ext.coords.xy[1])
        # clus2_ext = cluster2.projectxz().exterior
        # minzclus2 = min(clus2_ext.coords.xy[1])
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

    def add_crystal_from_above(self, crystal, lodge=0):
        # drop a new crystal onto the cluster
        # file = open('current_job.out', 'a+')
        # print('agg agg crystal from above',file=file)

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

        close_crystals = [self._crystals(n) for n in which_close[0]]

        # see which crystals could actually intersect with the new crystal
        close_crystals = [x for x in close_crystals if x.projectxy().intersects(crystal.projectxy())]
        # print('close crystals',file=file)
        # close_crystals = [ x for x in self.crystals() if x.projectxy().intersects(newpoly) ]
        if len(close_crystals) == 0:
            return False  # the crystal missed!

        # we know highest hit is >= max(minzs), therefore the first
        # hit can't be below (max(minzs) - height(crystal))
        minzs = [crystal2.minz for crystal2 in close_crystals]
        first_hit_lower_bound = max(minzs) - (crystal.maxz - crystal.minz)
        # remove the low crystals, sort from highest to lowest
        close_crystals = [x for x in close_crystals if x.maxz > first_hit_lower_bound]
        close_crystals.sort(key=lambda x: x.maxz, reverse=True)

        # look to see where the new crystal hits the old ones
        mindiffz = crystal.minz - first_hit_lower_bound  # the largest it can possibly be

        for crystal2 in close_crystals:

            if first_hit_lower_bound > crystal2.maxz:
                break  # stop looping if the rest of the crystals are too low
            diffz = crystal._min_vert_dist(crystal2)
            if diffz is None:
                break

            # return diffz
            # update if needed
            if diffz < mindiffz:
                mindiffz = diffz
                first_hit_lower_bound = crystal.minz - mindiffz
                # take the highest hit, move the crystal to that level
        crystal.move([0, 0, -mindiffz - lodge])

        return True

    def _min_vert_dist(self, crystal2):
        # find the minimum directed distance to crystal2 traveling straight downward
        rel_area = self.projectxy().buffer(0).intersection(crystal2.projectxy().buffer(0))
        # print(rel_area)
        if not isinstance(rel_area, geom.Polygon):
            print(None)
            return None
        c1_bottom = self.bottom()
        c2_top = crystal2.top()
        mindiffz = self.maxz - crystal2.minz

        # 1) lines and lines
        # all the intersections are calculated in 2d so no need to
        # convert these 3d objects!
        c1_lines = [l for l in c1_bottom['lines'] if l.intersects(rel_area)]
        c2_lines = [l for l in c2_top['lines'] if l.intersects(rel_area)]
        for line1 in c1_lines:
            for line2 in c2_lines:
                if line1.intersects(line2):
                    # get (2D) point of intersection
                    xy = line1.intersection(line2)
                    if not isinstance(xy, geom.point.Point):
                        # parallel lines don't count
                        continue
                    # get z difference
                    # make sure the damn lines aren't vertical
                    xrange1 = line1.xy[0][1] - line1.xy[0][0]
                    xrange2 = line2.xy[0][1] - line2.xy[0][0]
                    if xrange1 != 0:
                        # interpolate using x value
                        z1 = line1.interpolate((xy.x - line1.xy[0][0]) / (xrange1), normalized=True).z
                    else:
                        # interpolate using y value
                        z1 = line1.interpolate((xy.y - line1.xy[1][0]) / (line1.xy[1][1] - line1.xy[1][0]),
                                               normalized=True).z
                    if xrange2 != 0:
                        z2 = line2.interpolate((xy.x - line2.xy[0][0]) / (xrange2), normalized=True).z
                    else:
                        z2 = line2.interpolate((xy.y - line2.xy[1][0]) / (line2.xy[1][1] - line2.xy[1][0]),
                                               normalized=True).z
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz

        # 2) points and surfaces
        c1_points = [p for p in c1_bottom['points'] if p.intersects(rel_area)]
        c2_faces = [f for f in c2_top['faces'] if f.intersects(rel_area)]
        for point in c1_points:
            for face in c2_faces:
                if point.intersects(face):
                    # get z difference
                    z1 = point.z
                    # find the equation of the polygon's plane, plug in xy
                    a = np.array(face.exterior.coords[0])
                    AB = np.array(face.exterior.coords[1]) - a
                    AC = np.array(face.exterior.coords[2]) - a
                    normal_vec = np.cross(AB, AC)
                    # find constant value
                    d = -np.dot(normal_vec, a)
                    z2 = -(point.x * normal_vec[0] + point.y * normal_vec[1] + d) / normal_vec[2]
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz
                    # the point can only intersect one face, so we're
                    # done with this one
                    # break
                    # ^ I should be able to do that but I have to fix my 'bottom' function first!

        # 3) surfaces and points
        c1_faces = [f for f in c1_bottom['faces'] if f.intersects(rel_area)]
        c2_points = [p for p in c2_top['points'] if p.intersects(rel_area)]
        for point in c2_points:
            for face in c1_faces:
                if point.intersects(face):
                    # get z difference
                    z2 = point.z  # z2 this time!!!
                    # find the equation of the polygon's plane, plug in xy
                    a = np.array(face.exterior.coords[0])
                    AB = np.array(face.exterior.coords[1]) - a
                    AC = np.array(face.exterior.coords[2]) - a
                    normal_vec = np.cross(AB, AC)
                    # find constant value
                    d = -np.dot(normal_vec, a)
                    z1 = -(point.x * normal_vec[0] + point.y * normal_vec[1] + d) / normal_vec[2]
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz
                        # the point can only intersect one face, so we're
                        # done with this one
                    # break

        return mindiffz

    def _bottom(self):
        # return geometry of bottom side of falling crystal
        # to be used in connecting bottom of one crystal to the top of the other
        # getting the same points regardless of the orientation

        '''
        #Need to find lowest crystal that makes up the agg for intersection
        if self.ncrystals == 1:
            points = [ geom.Point(x) for x in self.points ]
            lowagg = min(points)
            if lowagg in points:
                print(True)
                self.points = self.points[x]

        else:
            points = self.points
        '''

        lines = []
        faces = []

        p0 = self.points[0]
        p6 = self.points[6]
        if abs(p0['x'] - p6['x']) < self.tol and abs(p0['y'] - p6['y']) < self.tol:
            # if it's vertical, only return the hexagon faces
            # (for now)
            for hexagon in range(2):
                n0 = hexagon * 6
                for i in range(5):
                    n = n0 + i
                    lines.append(geom.LineString([self.points[n], self.points[n + 1]]))
                lines.append(geom.LineString([self.points[n0 + 5], self.points[n0]]))
            # get the hexagons only-- no rectangles
            for n in range(2):
                i = n * 6
                faces.append(geom.Polygon(list(self.points[i:(i + 6)])))
        elif abs(p0['z'] - p6['z']) < self.tol:
            # lying flat on its side-- not returning hexagon faces
            if len(np.unique(self.points['z'])) == 4:
                # It's rotated so that there's a ridge on the top, and
                # the sides are vertical. Don't return any vertical
                # rectangular sides
                for n in range(5):
                    p1 = self.points[n]
                    p2 = self.points[n + 1]
                    # is it a non-vertical rectangle?
                    if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                        faces.append(geom.Polygon([self.points[n], self.points[n + 1],
                                                   self.points[n + 7], self.points[n + 6]]))
                # get that last rectangle missed
                p1 = self.points[5]
                p2 = self.points[0]
                if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                    faces.append(geom.Polygon([self.points[5], self.points[0],
                                               self.points[6], self.points[11]]))
                # get the lines around the hexagons
                for hexagon in range(2):
                    n0 = hexagon * 6
                    for i in range(5):
                        n = n0 + i
                        p1 = self.points[n]
                        p2 = self.points[n + 1]
                        if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                            lines.append(geom.LineString([self.points[n], self.points[n + 1]]))
                    p1 = self.points[n0 + 5]
                    p2 = self.points[n0]
                    if abs(p1['x'] - p2['x']) >= self.tol and abs(p1['y'] - p2['y']) >= self.tol:
                        lines.append(geom.LineString([self.points[n0 + 5], self.points[n0]]))
                # get the between-hexagon lines
                for n in range(6):
                    lines.append(geom.LineString([self.points[n], self.points[n + 6]]))

            # returning only rectangles
            pass
        else:
            # return all the faces

            # get the lines around the hexagons
            for hexagon in range(2):
                n0 = hexagon * 6
                for i in range(5):
                    n = n0 + i
                    lines.append(geom.LineString([self.points[n], self.points[n + 1]]))
                lines.append(geom.LineString([self.points[n0 + 5], self.points[n0]]))
            # get the between-hexagon lines
            for n in range(6):
                lines.append(geom.LineString([self.points[n], self.points[n + 6]]))
            # get the hexagons
            for n in range(2):
                i = n * 6
                faces.append(geom.Polygon(list(self.points[i:(i + 6)])))
            # get the rectangles
            for n in range(5):
                faces.append(geom.Polygon([self.points[n], self.points[n + 1],
                                           self.points[n + 7], self.points[n + 6]]))
            # get that last rectangle I missed
            faces.append(geom.Polygon([self.points[5], self.points[0],
                                       self.points[6], self.points[11]]))

        # return the geometry representing the bottom side of the prism

        # # similar to projectxy
        # if self.rotation[1] == math.pi / 2:
        #     # it's vertical, so just return one of the hexagons
        #     points = self.points[0:6]

        # first find top and bottom hexagon

        # remove the top two points

        # make the lines

        # make the faces

        return {'lines': lines, 'points': self.points, 'faces': faces}

    def _top(self):
        top = self.bottom()

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

    def _get_moments(self, poly):
        # get 'mass moments' for this cluster's 2D polygon using a
        # variation of the shoelace algorithm
        xys = poly.exterior.coords.xy
        npoints = len(xys[0])
        # values for the three points-- point[n], point[n+1], and
        # (0,0)-- making up triangular slices from the origin to the
        # edges of the polygon
        xmat = np.array([xys[0][0:-1], xys[0][1:], np.zeros(npoints - 1)]).transpose()
        ymat = np.array([xys[1][0:-1], xys[1][1:], np.zeros(npoints - 1)]).transpose()
        # arrange the points in left-center-right order
        x_order = np.argsort(xmat, axis=1)
        ordered_xmat = xmat[np.array([range(npoints - 1)]).transpose(), x_order]
        ordered_ymat = ymat[np.array([range(npoints - 1)]).transpose(), x_order]
        xl = ordered_xmat[:, 0]
        xm = ordered_xmat[:, 1]
        xr = ordered_xmat[:, 2]
        yl = ordered_ymat[:, 0]
        ym = ordered_ymat[:, 1]
        yr = ordered_ymat[:, 2]
        # which slices have areas on the left and right sides of the
        # middle point? Ignore values smaller than 'tol' so we don't
        # run into terrible problems with division.
        left = xm - xl > self.tol_ellipse
        right = xr - xm > self.tol_ellipse
        # slope and intercept of line connecting left and right points
        has_area = xr != xl
        m3 = np.zeros(npoints - 1)
        m3[has_area] = (yr[has_area] - yl[has_area]) / (xr[has_area] - xl[has_area])
        b3 = -xl * m3 + yl
        # the y coordinate of the line connecting the left and right
        # points at the x position of the middle point
        m3_mid = yl + m3 * (xm - xl)
        # is the midpoint above or below that line?
        mid_below = ym < m3_mid
        # line connecting left and middle point (where applicable)
        m1 = (ym[left] - yl[left]) / (xm[left] - xl[left])
        b1 = -xl[left] * m1 + yl[left]
        # line connecting middle and right point (where applicable)
        m2 = (yr[right] - ym[right]) / (xr[right] - xm[right])
        b2 = -xr[right] * m2 + yr[right]
        # now that we have the points in a nice format + helpful
        # information we can calculate the integrals of the slices
        xx = np.zeros(npoints - 1)
        xy = np.zeros(npoints - 1)
        yy = np.zeros(npoints - 1)
        dxl = (xm[left] - xl[left])
        dx2l = (xm[left] ** 2 - xl[left] ** 2)
        dx3l = (xm[left] ** 3 - xl[left] ** 3)
        dx4l = (xm[left] ** 4 - xl[left] ** 4)
        dxr = (xr[right] - xm[right])
        dx2r = (xr[right] ** 2 - xm[right] ** 2)
        dx3r = (xr[right] ** 3 - xm[right] ** 3)
        dx4r = (xr[right] ** 4 - xm[right] ** 4)
        # x^2
        xx[left] = dx4l * (m1 - m3[left]) / 4 + \
                   dx3l * (b1 - b3[left]) / 3
        xx[right] += dx4r * (m2 - m3[right]) / 4 + \
                     dx3r * (b2 - b3[right]) / 3
        # x*y
        xy[left] = dx4l * (m1 ** 2 - m3[left] ** 2) / 8 + \
                   dx3l * (b1 * m1 - b3[left] * m3[left]) / 3 + \
                   dx2l * (b1 ** 2 - b3[left] ** 2) / 4
        xy[right] += dx4r * (m2 ** 2 - m3[right] ** 2) / 8 + \
                     dx3r * (b2 * m2 - b3[right] * m3[right]) / 3 + \
                     dx2r * (b2 ** 2 - b3[right] ** 2) / 4
        # y^2
        yy[left] = dx4l * (m1 ** 3 - m3[left] ** 3) / 12 + \
                   dx3l * (b1 * m1 ** 2 - b3[left] * m3[left] ** 2) / 3 + \
                   dx2l * (b1 ** 2 * m1 - b3[left] ** 2 * m3[left]) / 2 + \
                   dxl * (b1 ** 3 - b3[left] ** 3) / 3
        yy[right] += dx4r * (m2 ** 3 - m3[right] ** 3) / 12 + \
                     dx3r * (b2 * m2 ** 2 - b3[right] * m3[right] ** 2) / 3 + \
                     dx2r * (b2 ** 2 * m2 - b3[right] ** 2 * m3[right]) / 2 + \
                     dxr * (b2 ** 3 - b3[right] ** 3) / 3
        # if the middle point was below the other points, multiply by
        # minus 1
        xx[mid_below] *= -1
        xy[mid_below] *= -1
        yy[mid_below] *= -1
        # find out which slices were going clockwise, and make those
        # negative
        points = np.array([xys[0], xys[1]]).transpose()
        cross_prods = np.cross(points[:-1], points[1:])
        clockwise = cross_prods < 0
        xx[clockwise] *= -1
        xy[clockwise] *= -1
        yy[clockwise] *= -1
        # add up the totals across the entire polygon
        xxtotal = np.sum(xx)
        yytotal = np.sum(yy)
        xytotal = np.sum(xy)
        # and if the points were in clockwise order, flip the sign
        if np.sum(cross_prods) < 0:
            xxtotal *= -1
            yytotal *= -1
            xytotal *= -1
        # also need to account for the holes, if they exist
        for linestring in list(poly.interiors):
            hole = geom.Polygon(linestring)
            hole_moments = self._get_moments(hole)
            xxtotal -= hole_moments[0]
            yytotal -= hole_moments[1]
            xytotal -= hole_moments[2]
        return [xxtotal, yytotal, xytotal]

    def _mvee(self, tol=0.01):  # mve = minimum volume ellipse
        # Based on work by Nima Moshtagh
        # http://www.mathworks.com/matlabcentral/fileexchange/9542

        """
        Finds the ellipse equation in "center form"
        (x-c).T * A * (x-c) = 1
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        points_arr = np.concatenate(self.points)[:self.ncrystals * 12]
        # print('points_Arr', points_arr)
        points_arr = np.array([list(i) for i in points_arr])
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

    def spheroid_axes(self, plates):
        A, c = self._mvee()
        U, D, V = la.svd(A)  # singular-value decomposition
        rx, ry, rz = 1. / np.sqrt(D)  # D is a diagonal matrix
        self.a, self.b, self.c = sorted([rx, ry, rz])

        if plates:
            self.agg_phi = self.c / self.a
            self.agg_r = np.power((np.power(self.a, 2) * self.c), (1. / 3.))
        else:
            self.agg_phi = self.a / self.c
            self.agg_r = np.power((np.power(self.c, 2) * self.a), (1. / 3.))

        return self.a, self.b, self.c

    def ellipse(self, u, v, rx, ry, rz):
        x = rx * np.cos(u) * np.cos(v)
        y = ry * np.sin(u) * np.cos(v)
        z = rz * np.sin(v)

        return x, y, z

    def plot_ellipsoid(self):

        A, centroid = self._mvee()
        # print('centroid', centroid)
        U, D, V = la.svd(A)
        # print(U, D, V)
        rx, ry, rz = 1. / np.sqrt(D)

        u, v = np.mgrid[0:2 * np.pi:20j, -np.pi / 2:np.pi / 2:10j]

        Ve = 4. / 3. * rx * ry * rz
        # print(Ve)

        E = np.dstack(self.ellipse(u, v, rx, ry, rz))

        E = np.dot(E, V) + centroid

        xell, yell, zell = np.rollaxis(E, axis=-1)

        x = np.zeros((len(self.points['x']), 27))
        y = np.zeros((len(self.points['x']), 27))
        z = np.zeros((len(self.points['x']), 27))

        X = self.points['x']
        Y = self.points['y']
        Z = self.points['z']

        Xlim = self.points['x'][:self.ncrystals]
        Ylim = self.points['y'][:self.ncrystals]
        Zlim = self.points['z'][:self.ncrystals]
        # for i in range(0, 360, 60):
        #    print('angle', i)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        # 90, 0 for z orientation, 0, 90 for y orientation, 0, 0 for x orientation
        # ax.view_init(elev=90, azim=270)
        ax.view_init(elev=0, azim=90)
        ax.plot_surface(xell, yell, zell, cstride=1, rstride=1, alpha=0.2)

        data = []
        # print(self.ncrystals)
        for l in range(self.ncrystals):

            prismind = [0, 6, 7, 1, 2, 8, 9, 3, 4, 10, 11, 5]  # prism lines
            i = 0
            for n in prismind:
                x[l][i] = X[l][n]
                y[l][i] = Y[l][n]
                z[l][i] = Z[l][n]
                i += 1

            if l == len(self.points['x'][:self.ncrystals]) - 1:
                color = 'k'
            else:
                color = 'b'

            ax.plot(x[l][0:12], y[l][0:12], z[l][0:12], color=color)

            i = 0
            for n in range(0, 6):  # basal face lines

                x[l][i + 12] = X[l][n]
                y[l][i + 12] = Y[l][n]
                z[l][i + 12] = Z[l][n]
                i += 1

            x[l][18] = X[l][0]
            y[l][18] = Y[l][0]
            z[l][18] = Z[l][0]

            ax.plot(x[l][12:19], y[l][12:19], z[l][12:19], color=color)

            i = 0
            for n in range(6, 12):  # basal face lines

                x[l][i + 19] = X[l][n]
                y[l][i + 19] = Y[l][n]
                z[l][i + 19] = Z[l][n]
                i += 1

            x[l][25] = X[l][6]
            y[l][25] = Y[l][6]
            z[l][25] = Z[l][6]

            ax.plot(x[l][19:26], y[l][19:26], z[l][19:26], color=color)

            maxX = np.max(Xlim)
            minX = np.min(Xlim)
            maxY = np.max(Ylim)
            minY = np.min(Ylim)
            maxZ = np.max(Zlim)
            minZ = np.min(Zlim)

            maxXe = np.max(xell)
            minXe = np.min(xell)
            maxYe = np.max(yell)
            minYe = np.min(yell)
            maxZe = np.max(zell)
            minZe = np.min(zell)

            maxxyz = max(maxX, maxY, maxZ)
            minxyz = min(minX, minY, minZ)

            minell = min(minXe, minYe, minZe)
            maxell = max(maxXe, maxYe, maxZe)
            # print('min',minell, maxell)
            ax.set_xlim(minxyz, maxxyz)
            ax.set_ylim(minxyz, maxxyz)
            ax.set_zlim(minxyz, maxxyz)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # ax.set_zticklabels([])
            # ax.view_init(30, i)
            ax.view_init(0, 90)
            # plt.pause(.001)
        plt.show()
        # fig.savefig(fname='ellipsoid_columns.eps')

    def plot_ellipsoid_agg_agg(self, cluster, nearest_geoms, nearest_geoms_y, view):

        A, centroid = self._mvee()
        # print('centroid', centroid)
        U, D, V = la.svd(A)
        # print(U, D, V)
        rx, ry, rz = 1. / np.sqrt(D)

        u, v = np.mgrid[0:2 * np.pi:20j, -np.pi / 2:np.pi / 2:10j]

        Ve = 4. / 3. * rx * ry * rz
        # print(Ve)

        E = np.dstack(self.ellipse(u, v, rx, ry, rz))

        E = np.dot(E, V) + centroid

        xell, yell, zell = np.rollaxis(E, axis=-1)

        x = np.zeros((len(self.points['x']), 27))
        y = np.zeros((len(self.points['x']), 27))
        z = np.zeros((len(self.points['x']), 27))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        # 90, 0 for z orientation, 0, 90 for y orientation, 0, 0 for x orientation
        # ax.view_init(elev=90, azim=270)
        print(view)
        if view == 'x':
            ax.view_init(elev=0, azim=90)
        elif view == 'y':
            ax.view_init(elev=0, azim=0)
        else:
            ax.view_init(elev=90, azim=180)
        ax.plot_surface(xell, yell, zell, cstride=1, rstride=1, alpha=0.2, zorder=-1)
        # clusters = [self, cluster2]
        # for i in clusters:

        # color = 'orange'
        X = self.points['x']
        Y = self.points['y']
        Z = self.points['z']

        Xlim = self.points['x']
        Ylim = self.points['y']
        Zlim = self.points['z']
        # for i in range(0, 360, 60):
        #    print('angle', i)
        data = []
        print('ncrys', self.ncrystals, cluster.ncrystals)
        for l in range(self.ncrystals):

            if l < self.ncrystals - cluster.ncrystals:
                color = 'r'
            else:
                color = 'k'

            prismind = [0, 6, 7, 1, 2, 8, 9, 3, 4, 10, 11, 5]  # prism lines
            i = 0
            for n in prismind:
                x[l][i] = X[l][n]
                y[l][i] = Y[l][n]
                z[l][i] = Z[l][n]
                i += 1

            # if l == len(self.points['x'][:self.ncrystals])-1:
            #    color = 'white'
            # else:
            #    color = 'orange'

            ax.plot(x[l][0:12], y[l][0:12], z[l][0:12], color=color)

            i = 0
            for n in range(0, 6):  # basal face lines

                x[l][i + 12] = X[l][n]
                y[l][i + 12] = Y[l][n]
                z[l][i + 12] = Z[l][n]
                i += 1

            x[l][18] = X[l][0]
            y[l][18] = Y[l][0]
            z[l][18] = Z[l][0]

            ax.plot(x[l][12:19], y[l][12:19], z[l][12:19], color=color)

            i = 0
            for n in range(6, 12):  # basal face lines

                x[l][i + 19] = X[l][n]
                y[l][i + 19] = Y[l][n]
                z[l][i + 19] = Z[l][n]
                i += 1

            x[l][25] = X[l][6]
            y[l][25] = Y[l][6]
            z[l][25] = Z[l][6]

            ax.plot(x[l][19:26], y[l][19:26], z[l][19:26], color=color)

            maxX = np.max(Xlim)
            minX = np.min(Xlim)
            maxY = np.max(Ylim)
            minY = np.min(Ylim)
            maxZ = np.max(Zlim)
            minZ = np.min(Zlim)

            maxXe = np.max(xell)
            minXe = np.min(xell)
            maxYe = np.max(yell)
            minYe = np.min(yell)
            maxZe = np.max(zell)
            minZe = np.min(zell)

            maxxyz = max(maxX, maxY, maxZ)
            minxyz = min(minX, minY, minZ)

            minell = min(minXe, minYe, minZe)
            maxell = max(maxXe, maxYe, maxZe)
            # print('min',minell, maxell)
            ax.set_xlim(minxyz, maxxyz)
            ax.set_ylim(minxyz, maxxyz)
            ax.set_zlim(minxyz, maxxyz)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ##ax.set_zticklabels([])
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            # ax.grid(False)

            # ax.view_init(30, i)
            # plt.pause(.001)

        if view == 'x':
            ax.scatter(nearest_geoms[0].x, nearest_geoms_y[0].y, nearest_geoms[0].y, c='red', s=100, zorder=10)
            ax.scatter(nearest_geoms[1].x, nearest_geoms_y[1].y, nearest_geoms[1].y, c='k', s=100, zorder=10)
        elif view == 'y':
            ax.scatter(nearest_geoms[0].x, nearest_geoms_y[0].x, nearest_geoms_y[0].y, c='red', s=100, zorder=10)
            ax.scatter(nearest_geoms[1].x, nearest_geoms_y[1].x, nearest_geoms_y[1].y, c='k', s=100, zorder=10)

        # fig.savefig('plot_ellipsoids/ellipse.eps',rasterized=True)

        plt.show()

    def plot_constraints(self, agg_pt, new_pt, plates, new_crystal, k, plot_dots=plot):

        # fig = plt.figure(1, figsize=(5,5))
        # ax = fig.add_subplot(111)
        fig, ax = plt.subplots(1, 1)
        area = []
        centroid = []

        for i in range(2):

            if i == 0:
                color = '#29568F'
                zorder = 3
                linecolor = '#29568F'
                projpoly = self.projectxy()  # agg
            else:
                zorder = 2
                color = '#e65c00'  # orange
                linecolor = '#e65c00'

                # projpoly = geom.MultiPoint(self.points[self.ncrystals][['x', 'y']]).convex_hull
                projpoly = new_crystal.projectxy()

                rel_area = self.projectxy().buffer(0).intersection(projpoly.buffer(0))
                ovrlpptch = PolygonPatch(rel_area, fill=True, ec='k', fc='k', zorder=3)
                ax.add_patch(ovrlpptch)
                # xovrlp,yovrlp = list(rel_area.exterior.coords.xy)
                # ax.plot(xovrlp, yovrlp, 'o',color ='green', linewidth = 3, zorder =4)

            area.append(projpoly.area)
            centroid.append(projpoly.centroid)
            cryspatch = PolygonPatch(projpoly, fill=True, ec='k', fc=color, zorder=zorder, alpha=1.0)
            ax.add_patch(cryspatch)

        crystals1 = [self, new_crystal]  # self is the prexisting clus
        d = []
        for i in crystals1:
            x, y = np.array(list(i.projectxy().exterior.coords.xy))
            maxx = max(x)
            minx = min(x)
            maxy = max(y)
            miny = min(y)

            maxy_atmaxx = y[np.where(x == maxx)]
            maxx_atmaxy = x[np.where(y == maxy)]
            minx_atminy = x[np.where(y == miny)]
            miny_atminx = y[np.where(x == minx)]

            dmaxx = Point(maxx, maxy_atmaxx).distance(Point(minx, miny_atminx))
            dmaxy = Point(maxx_atmaxy, maxy).distance(Point(miny_atminx, miny))
            d.append(max(dmaxx, dmaxy))

            ax.plot([minx, maxx], [miny_atminx, maxy_atmaxx], color='w', linewidth=3, zorder=8)
            # ax.plot([minx_atminy, maxx_atmaxy], [miny, maxy], color ='w', linewidth = 3, zorder=8)
            ax.plot(x, y, 'o', color='w', linewidth=3, zorder=11)

        l = (self.projectxy().centroid).distance(new_crystal.projectxy().centroid)
        S = 2 * l / (d[0] + d[1])

        if plates:
            lmax = 0.6 * (d[0] + d[1]) / 2  # S parameter can't be higher than 0.6 for plates
        else:
            lmax = 0.3 * (d[0] + d[1]) / 2  # S parameter can't be higher than 0.3 for columns

        lmax_bound = self.projectxy().buffer(0).centroid.buffer(lmax)
        # new crystal center can't be outside of this circle
        ax.add_patch(descartes.PolygonPatch(lmax_bound, fc='gray', ec='k', alpha=0.3, zorder=8))

        ax.plot([centroid[0].x, centroid[1].x], [centroid[0].y, centroid[1].y], color='gold', linewidth=4, zorder=8)

        if plot_dots:
            i = 0
            while i < 1000:
                angle = random.uniform(0, 1) * np.pi * 2
                radius = np.sqrt(random.uniform(0, 1)) * lmax
                originX = lmax_bound.centroid.xy[0]
                originY = lmax_bound.centroid.xy[1]

                x = originX + radius * np.cos(angle)
                y = originY + radius * np.sin(angle)
                ax.scatter(x[0], y[0], color='g', s=10, zorder=10)
                i += 1

        xmin = min(lmax_bound.exterior.coords.xy[0])
        xmax = max(lmax_bound.exterior.coords.xy[0])
        ymin = min(lmax_bound.exterior.coords.xy[1])
        ymax = max(lmax_bound.exterior.coords.xy[1])
        # print(xmin, xmax, ymin, ymax)
        # print(centroid[1])
        square = geom.Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])

        # ax.add_patch(descartes.PolygonPatch(square, fc='gray', ec='k', alpha=0.1))

        ax.axis('scaled')
        if plot_dots:
            path = ('/Users/vprzybylo/Desktop/ovrlp_constraint' + str(k) + '_dots' + '.pdf')
        else:
            path = ('/Users/vprzybylo/Desktop/ovrlp_constraint' + str(k) + '.pdf')

    # plt.savefig(path)

    # plt.show()

    def plot_constraints_accurate(self, agg_pt, new_pt, plates, new_crystal, k, plot_dots=False):

        # fig = plt.figure(1, figsize=(5,5))
        # ax = fig.add_subplot(111)
        fig, ax = plt.subplots(1, 1)
        area = []
        centroid = []
        dmax1 = []
        dmax2 = []

        for i in range(2):

            if i == 0:
                color = '#29568F'
                zorder = 3
                linecolor = '#29568F'
                projpoly = self.projectxy()

            else:
                zorder = 2
                color = '#e65c00'  # orange
                linecolor = '#e65c00'

                # projpoly = geom.MultiPoint(self.points[self.ncrystals][['x', 'y']]).convex_hull
                projpoly = new_crystal.projectxy()

                rel_area = self.projectxy().buffer(0).intersection(projpoly.buffer(0))
                # ovrlpptch = PolygonPatch(rel_area, fill=True, ec='k', fc='k', zorder=3)
                # ax.add_patch(ovrlpptch)
                # xovrlp,yovrlp = list(rel_area.exterior.coords.xy)
                # ax.plot(xovrlp, yovrlp, 'o',color ='green', linewidth = 3, zorder =4)

            area.append(projpoly.area)
            centroid.append(projpoly.centroid)
            cryspatch = PolygonPatch(projpoly, fill=True, ec=color, fc=color, zorder=zorder, alpha=1.0)
            ax.add_patch(cryspatch)

            x, y = list(projpoly.exterior.coords.xy)

            dinit = 0
            for j in range(len(x)):
                for l in range(len(x)):
                    d = (Point(x[l], y[l]).distance(Point(x[j], y[j])))
                    if d > dinit:
                        dinit = d

                        if i == 0:
                            dmax1.append(d)
                        if i == 1:
                            dmax2.append(d)
                        xstart = l
                        ystart = l
                        xend = j
                        yend = j

            if i == 0:
                dmax1 = max(dmax1)

            if i == 1:
                dmax2 = max(dmax2)

            ax.plot([x[xstart], x[xend]], [y[ystart], y[yend]], color='w', linewidth=3, zorder=8, alpha=0.5)

            ax.plot(x, y, 'o', color='w', linewidth=3, zorder=11, alpha=0.5)

        l = centroid[0].distance(centroid[1])
        S = 2 * l / (dmax1 + dmax2)

        if plates:
            lmax = 0.6 * (dmax1 + dmax2) / 2  # force S = .6 for plates
        else:
            lmax = 0.3 * (dmax1 + dmax2) / 2  # force S = .3 for columns

        lmax_bound = self.projectxy().centroid.buffer(lmax)
        # new crystal center can't be outside of this circle
        ax.add_patch(descartes.PolygonPatch(lmax_bound, fc='gray', ec='k', alpha=0.3, zorder=8))

        ax.plot([centroid[0].x, centroid[1].x], [centroid[0].y, centroid[1].y], color='gold', linewidth=4, zorder=8)

        if plot_dots:
            i = 0
            while i < 1000:
                angle = random.uniform(0, 1) * np.pi * 2
                radius = np.sqrt(random.uniform(0, 1)) * lmax
                originX = lmax_bound.centroid.xy[0]
                originY = lmax_bound.centroid.xy[1]

                x = originX + radius * np.cos(angle)
                y = originY + radius * np.sin(angle)
                ax.scatter(x[0], y[0], color='g', s=10, zorder=10)

                i += 1

        # nmisses = 100
        # n = 0
        # while n < nmisses:
        # if centroid[1].within(lmax_bound):

        xmin = min(lmax_bound.exterior.coords.xy[0])
        xmax = max(lmax_bound.exterior.coords.xy[0])
        ymin = min(lmax_bound.exterior.coords.xy[1])
        ymax = max(lmax_bound.exterior.coords.xy[1])
        # print(xmin, xmax, ymin, ymax)
        # print(centroid[1])
        square = geom.Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
        plt.plot(agg_pt.x, agg_pt.y, 'darkblue', marker='o', markersize=8, zorder=11)
        plt.plot(new_pt.x, new_pt.y, 'red', marker='o', markersize=8, zorder=11)
        # break
        # else:
        # n+=1
        # ax.add_patch(descartes.PolygonPatch(square, fc='gray', ec='k', alpha=0.1))

        ax.axis('scaled')
        if plot_dots:
            path = ('/Users/vprzybylo/Desktop/ovrlp_constraint' + str(k) + '_dots' + '.pdf')
        else:
            path = ('/Users/vprzybylo/Desktop/ovrlp_constraint' + str(k) + '.pdf')

        # plt.savefig(path)

        plt.show()

    def fit_ellipse(self, dims):
        # Emulating this function, but for polygons in continuous
        # space rather than blobs in discrete space:
        # http://www.idlcoyote.com/ip_tips/fit_ellipse.html

        if dims == [['x', 'y']]:
            try:
                poly = self.projectxy()
            except ValueError:
                return None
        if dims == [['x', 'z']]:
            try:
                poly = self.projectxz()
            except ValueError:
                return None
        if dims == [['y', 'z']]:
            try:
                poly = self.projectyz()
            except ValueError:
                return None

        xy_area = poly.area

        # center the polygon around the centroid
        centroid = poly.centroid
        poly = sha.translate(poly, -centroid.x, -centroid.y)

        # occasionally we get multipolygons
        if isinstance(poly, geom.MultiPolygon):
            xx = 0
            yy = 0
            xy = 0
            for poly2 in poly:
                moments = self._get_moments(poly2)
                xx += moments[0] / xy_area
                yy += moments[1] / xy_area
                xy -= moments[2] / xy_area
        else:
            moments = self._get_moments(poly)
            xx = moments[0] / xy_area
            yy = moments[1] / xy_area
            xy = -moments[2] / xy_area

        # get fit ellipse axes lengths, orientation, center
        m = np.matrix([[yy, xy], [xy, xx]])
        evals, evecs = np.linalg.eigh(m)
        semimajor = np.sqrt(evals[0]) * 2
        semiminor = np.sqrt(evals[1]) * 2
        major = semimajor * 2
        minor = semiminor * 2
        # print('semi', semimajor, evals)

        evec = np.squeeze(np.asarray(evecs[0]))
        orientation = np.arctan2(evec[1], evec[0]) * 180 / np.pi

        ellipse = {'xy': [centroid.x, centroid.y], 'width': minor,
                   'height': major, 'angle': orientation}
        # print('crystals',self.ncrystals)
        # print('ell',ellipse['height'])
        return ellipse

    def plot_ellipse(self, dims):

        # Only (x,z), (y,z), and (x,y) needed/allowed for dimensions
        # Depth works for both side views (x,z) and (y,z)

        if dims == [['x', 'z']]:
            # self.rotate_to([np.pi / 2, 0, 0])
            poly = self.projectxz()
        elif dims == [['y', 'z']]:
            # self.rotate_to([np.pi / 2, np.pi / 2, 0])
            poly = self.projectyz()
        elif dims == [['x', 'y']]:  # this is the only projection used in the aggregate aspect ratio calculation
            # self.rotate_to([0, 0, 0])
            poly = self.projectxy()
        else:
            print('Not a valid dimension')

        params = self.fit_ellipse(dims)
        ellipse = Ellipse(**params)

        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        ax.add_artist(ellipse)
        ellipse.set_alpha(.9)  # opacity
        ellipse.set_facecolor('darkorange')
        # if isinstance(poly, geom.multipolygon.MultiPolygon):
        #    for poly2 in poly:
        #        x, y = poly2.exterior.xy
        # ax.plot(x, y, color = 'green', linewidth = 3)
        # else:
        #    x, y = poly.exterior.xy
        # ax.plot(x, y, color = 'green', linewidth = 3)

        # maxdim = max([params['width'], params['height']]) / 2
        # ax.set_xlim([-maxdim + params['xy'][0], maxdim + params['xy'][0]])
        # ax.set_ylim([-maxdim + params['xy'][1], maxdim + params['xy'][1]])

        for l in range(len(dims)):

            crysmaxz = []
            crysminz = []
            maxzinds = []
            minzinds = []
            for i in range(self.ncrystals):
                hex1pts = self.points[dims[l]][i][0:6]  # first basal face
                poly1 = geom.Polygon([[p[0], p[1]] for p in hex1pts])  # make it into a polygon to plot
                hex2pts = self.points[dims[l]][i][6:12]  # second basal face
                poly2 = geom.Polygon([[p[0], p[1]] for p in hex2pts])
                x1, y1 = poly1.exterior.xy  # array of xy points
                x2, y2 = poly2.exterior.xy

                if i == 1:
                    color = 'navy'
                    zorder = 3
                else:
                    color = 'darkgreen'
                    zorder = 4
                for n in range(7):  # plot the prism face lines
                    x = [x1[n], x2[n]]
                    y = [y1[n], y2[n]]
                    ax.plot(x, y, color=color, zorder=zorder, linewidth='2')

                # polypatch1 = PolygonPatch(poly1, fill=True, zorder = 1)
                # polypatch2 = PolygonPatch(poly2, fill=True, zorder = 1)
                ax.plot(x1, y1, color=color, zorder=2, linewidth='2')  # edges of polygons
                ax.plot(x2, y2, color=color, zorder=4, linewidth='2')
                # ax.add_patch(polypatch1)
                # ax.add_patch(polypatch2)

                # for plotting depth line segment:
                crysminz.append(self.points['z'][i].min())
                crysmaxz.append(self.points['z'][i].max())
                minzinds.append(np.argmin(self.points['z'][i]))  # index of min pt for every xtal
                maxzinds.append(np.argmax(self.points['z'][i]))  # index of max pt

            maxcrysind, self.maxz = max(enumerate(crysmaxz), key=operator.itemgetter(1))  # overall max btwn xtals
            mincrysind, self.minz = min(enumerate(crysminz), key=operator.itemgetter(1))
            xdepthmin = self.points[dims[l]][mincrysind][minzinds[mincrysind]]
            xdepthmax = self.points[dims[l]][maxcrysind][maxzinds[maxcrysind]]

            # ax.plot(xdepthmin, self.minz, 'ko', linewidth = '4')
            # ax.plot(xdepthmax, self.maxz, 'ko', linewidth = '4')
            depthlinex = [0, 0]
            depthliney = [self.minz, self.maxz]
            # ax.plot(depthlinex,depthliney, 'k', linewidth = '4')

            ######## plot major and minor axes ############

            maxdim = max([params['width'], params['height']]) / 2  # major axis

            # ax.set_xlim([-maxdim + params['xy'][0], maxdim + params['xy'][0]])
            # ax.set_ylim([-maxdim + params['xy'][1], maxdim + params['xy'][1]])

            leftverticex = params['xy'][0] - params['width'] / 2
            leftverticey = params['xy'][1]
            rightverticex = params['xy'][0] + params['width'] / 2
            rightverticey = params['xy'][1]
            # plt.plot(leftverticex, leftverticey, 'ro', markersize = 5)  #original vertices if no angle
            # plt.plot(rightverticex, rightverticey, 'ro', markersize = 5)
            # plt.plot(params['xy'][0], params['xy'][1], 'wo', markersize = 7)

            radangle = params['angle'] * np.pi / 180
            # orientation angle of ellipse

            # rotate axis points and reposition if off center
            newxleft = ((leftverticex - params['xy'][0]) * np.cos(radangle) - \
                        (leftverticey - params['xy'][1]) * np.sin(radangle)) + params['xy'][0]

            newxright = ((rightverticex - params['xy'][0]) * np.cos(radangle) - \
                         (rightverticey - params['xy'][1]) * np.sin(radangle)) + params['xy'][0]

            newyleft = ((leftverticex - params['xy'][0]) * np.sin(radangle) + \
                        (leftverticey - params['xy'][1]) * np.cos(radangle)) + params['xy'][1]

            newyright = ((rightverticex - params['xy'][0]) * np.sin(radangle) + \
                         (rightverticey - params['xy'][1]) * np.cos(radangle)) + params['xy'][1]

            newx = [newxleft, newxright]
            newy = [newyleft, newyright]
            ax.plot(newx, newy, color='white', linewidth=3)  # major/minor axis lines
            ax.plot(newx, newy, 'wo', markersize=7)

            radangle1 = params['angle'] * np.pi / 180 + np.pi / 2
            radangle = radangle1
            leftverticex = params['xy'][0] - params['height'] / 2
            rightverticex = params['xy'][0] + params['height'] / 2

            newxleft = ((leftverticex - params['xy'][0]) * np.cos(radangle) - \
                        (leftverticey - params['xy'][1]) * np.sin(radangle)) + params['xy'][0]

            newxright = ((rightverticex - params['xy'][0]) * np.cos(radangle) - \
                         (rightverticey - params['xy'][1]) * np.sin(radangle)) + params['xy'][0]

            newyleft = ((leftverticex - params['xy'][0]) * np.sin(radangle) + \
                        (leftverticey - params['xy'][1]) * np.cos(radangle)) + params['xy'][1]

            newyright = ((rightverticex - params['xy'][0]) * np.sin(radangle) + \
                         (rightverticey - params['xy'][1]) * np.cos(radangle)) + params['xy'][1]

            newx = [newxleft, newxright]
            newy = [newyleft, newyright]
            ax.plot(newx, newy, color='white', linewidth=3)
            ax.plot(newx, newy, 'wo', markersize=2)

            ax.set_aspect('equal', 'datalim')
            plt.show()
        return params

    '''
    def write_obj(self, filename):
        f = open(filename, 'w')

        faces = []
        for i, crystal in enumerate(self.crystals()):
            nc = i * 12
            # write the vertices
            for n in range(12):
                f.write('v ' + ' '.join(map(str, crystal.points[n])) + '\n')
            # write the hexagons
            for n in range(2):
                coords = range(n * 6 + 1 + nc, (n + 1) * 6 + 1 + nc)
                faces.append('f ' + ' '.join(map(str, coords)))
            # write the rectangles
            for n in range(5):
                coords = [n + 1 + nc, n + 2 + nc, n + 8 + nc, n + 7 + nc]
                faces.append('f ' + ' '.join(map(str, coords)))
            # write the last rectangle I missed
            coords = [nc + 6, nc + 1, nc + 7, nc + 12]
            faces.append('f ' + ' '.join(map(str, coords)))
        f.write('\n'.join(faces))
        f.close()

    def intersect(self):
        from operator import itemgetter
        # return a multiline object representing the edges of the prism
        hex_cntmax = np.empty([self.ncrystals],dtype='object')
        hex_cntmin = np.empty([self.ncrystals],dtype='object')
        for c in range(self.ncrystals):
            # make a line connecting the two hexagons at the max x value
            dim = ['y','z']
            hex1pts = self.points[dim][c,0:6]
            hex2pts = self.points[dim][c,6:12]
            hex1max = max(self.points[c,0:6][dim],key=itemgetter(0))
            hex2max = max(self.points[c,6:12][dim],key=itemgetter(0))
            hex_cntmax[c] = geom.LineString((hex1max, hex2max))
            print(hex1pts)
            print(hex1max)
            print(hex2pts)
            print(hex2max)
            hex1min = min(self.points[c,0:6][dim],key=itemgetter(0))
            hex2min = min(self.points[c,6:12][dim],key=itemgetter(0))

            hex_cntmin[c] = geom.LineString((hex1min, hex2min))

        intersect = False

        max_intersect = hex_cntmax[0].intersects(hex_cntmax[1])
        min_intersect = hex_cntmin[0].intersects(hex_cntmin[1])
        minmax_intersect = hex_cntmin[0].intersects(hex_cntmax[1])
        maxmin_intersect = hex_cntmax[0].intersects(hex_cntmin[1])
        if max_intersect==True:
            intersect = True
            print('max')
        if min_intersect==True:
            intersect = True
            print('min')
        if minmax_intersect==True:
            intersect = True
            print('minmax')
        if maxmin_intersect == True:
            intersect = True
            print('maxmin')
        print(intersect)

        return intersect

    '''

    def aspect_ratio(self, cluster2, method, minor):
        # rotation = self.rotation

        # get depth measurement in z

        self.maxz = self.points['z'][:self.ncrystals].max()
        self.minz = self.points['z'][:self.ncrystals].min()
        # print(self.maxz, self.minz, self.maxz-self.minz)
        self.depth = self.maxz - self.minz

        # self.rotate_to([0, 0, 0]) #commented out to keep at current rotation
        # getting ellipse axes from 3 perspectives
        ellipse = {}
        dims = [['x', 'y']]
        ellipse['z'] = self.fit_ellipse(dims)
        dims = [['x', 'z']]
        # self.rotate_to([np.pi / 2, 0, 0])
        ellipse['y'] = self.fit_ellipse(dims)
        dims = [['y', 'z']]
        # self.rotate_to([np.pi / 2, np.pi / 2, 0])
        ellipse['x'] = self.fit_ellipse(dims)

        # put the cluster back
        # self.rotate_to([0, 0, 0])

        for dim in ellipse.keys():
            self.major_axis[dim] = max(ellipse[dim]['height'], ellipse[dim]['width'])
            self.minor_axis[dim] = min(ellipse[dim]['height'], ellipse[dim]['width'])

        if minor == 'minorxy':
            if method == 1:
                return max(self.major_axis.values()) / max(self.minor_axis.values())
            elif method == 'plate':
                return max(self.minor_axis['x'], self.minor_axis['y']) / self.major_axis['z']
            elif method == 'column':
                return self.major_axis['z'] / max(self.minor_axis['x'], self.minor_axis['y'])
        elif minor == 'depth':  # use depth as minor dimension of aggregate
            if method == 1:
                return max(self.major_axis.values()) / max(self.minor_axis.values())
            elif method == 'plate':
                # print(self.depth, self.major_axis['z'], self.depth/self.major_axis['z'])
                return self.depth / self.major_axis['z']
            elif method == 'column':
                # print(self.major_axis['z'], self.depth, self.major_axis['z']/self.depth)
                return self.major_axis['z'] / self.depth

    def phi_2D(self):

        ellipse = self.fit_ellipse([['x', 'z']])
        if ellipse is None:
            ellipse = self.fit_ellipse([['y', 'z']])
        if ellipse is None:
            print('returning None out of phi 2D')
            return None

        major_axis = max(ellipse['height'], ellipse['width'])
        minor_axis = min(ellipse['height'], ellipse['width'])

        return (minor_axis / major_axis)

    def phi_2D_rotate(self):
        # As in Jiang 2017, KOROLEV AND ISSAC 2003, and Garret 2015

        phi = []
        major_axis = {}
        minor_axis = {}
        for i in range(0, 108, 36):
            self.rotate_to([0, 0, i * 180 / np.pi])
            ellipse = self.fit_ellipse([['x', 'z']])
            if ellipse is None:
                ellipse = self.fit_ellipse([['y', 'z']])
            major_axis['y'] = max(ellipse['height'], ellipse['width'])
            minor_axis['y'] = min(ellipse['height'], ellipse['width'])
            phi.append((minor_axis['y']) / (major_axis['y']))
            # print(self.ncrystals, sum(phi)/len(phi), sum(phi), phi, ellipse, ellipse['height'], ellipse['width'], (minor_axis['y']) / (major_axis['y']), type(sum(phi)/len(phi)))
        if np.isnan(phi).any():
            self.plot_ellipse(dims=[['y', 'z']])
            self.plot_ellipse(dims=[['x', 'z']])
            self.plot_ellipse(dims=[['x', 'y']])

        self.phi2D = sum(phi) / len(phi)

        # if np.isnan(phi).any():

        #    print(self.ncrystals, sum(phi)/len(phi), sum(phi), phi, ellipse, ellipse['height'], ellipse['width'], (minor_axis['x']) / (major_axis['x']))

        return self.phi2D
        # return reduce(lambda x, y: x + y, phi) / len(phi)

    def depth_horizontal_phi(self):
        # "A Statistical and Physical Description
        # of Hydrometeor Distributions in Colorado Snowstorms
        # Using a Video Disdrometer” by Brandes
        maxx = np.nanmax(self.points['x'])
        minx = np.nanmin(self.points['x'])
        return (self.depth / (maxx - minx))

    def overlap(self, new_crystal, seedcrystal):

        agg_nonew = self.projectxy().buffer(0)

        rel_area = self.projectxy().buffer(0).intersection(new_crystal.projectxy().buffer(0))
        # rel_area = self.projectxy().buffer(0).intersection(new_crystal.projectxy().buffer(0))

        # pctovrlp1 = (rel_area.area/(seedcrystal.projectxy().area+new_crystal.projectxy().area-rel_area.area))*100
        pctovrlp = (rel_area.area / (self.projectxy().area + new_crystal.projectxy().area)) * 100
        # pctovrlp = (rel_area.area/(new_crystal.projectxy().area+self.projectxy().area))*100
        # print('rel',rel_area.area)
        # print(pctovrlp)
        return (pctovrlp)

    def _make_circle(self, points):
        # Convert to float and randomize order
        shuffled = [(float(x), float(y)) for (x, y) in points]
        random.shuffle(shuffled)

        # Progressively add points to circle or recompute circle
        c = None
        for (i, p) in enumerate(shuffled):
            if c is None or not self.is_in_circle(c, p):
                c = self._make_circle_one_point(shuffled[: i + 1], p)
        return c

    def _make_circle_one_point(self, points, p):
        # One boundary point known
        c = (p[0], p[1], 0.0)
        for (i, q) in enumerate(points):
            if not self.is_in_circle(c, q):
                if c[2] == 0.0:
                    c = self._make_diameter(p, q)
                else:
                    c = self._make_circle_two_points(points[: i + 1], p, q)
        return c

    def _make_circle_two_points(self, points, p, q):
        # Two boundary points known
        circ = self._make_diameter(p, q)
        left = None
        right = None
        px, py = p
        qx, qy = q

        # For each point not in the two-point circle
        for r in points:
            if self.is_in_circle(circ, r):
                continue

            # Form a circumcircle and classify it on left or right side
            cross = self._cross_product(px, py, qx, qy, r[0], r[1])
            c = self._make_circumcircle(p, q, r)
            if c is None:
                continue
            elif cross > 0.0 and (left is None or self._cross_product(px, py, qx, qy, c[0], c[1])
                                  > self._cross_product(px, py, qx, qy, left[0], left[1])):
                left = c
            elif cross < 0.0 and (right is None or self._cross_product(px, py, qx, qy, c[0], c[1])
                                  < self._cross_product(px, py, qx, qy, right[0], right[1])):
                right = c

        # Select which circle to return
        if left is None and right is None:
            return circ
        elif left is None:
            return right
        elif right is None:
            return left
        else:
            return left if (left[2] <= right[2]) else right

    def _make_circumcircle(self, p0, p1, p2):
        # Mathematical algorithm from Wikipedia: Circumscribed circle
        ax, ay = p0
        bx, by = p1
        cx, cy = p2
        ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
        oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
        ax -= ox;
        ay -= oy
        bx -= ox;
        by -= oy
        cx -= ox;
        cy -= oy
        d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
        if d == 0.0:
            return None
        x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
                    ay - by)) / d
        y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
                    bx - ax)) / d
        ra = math.hypot(x - p0[0], y - p0[1])
        rb = math.hypot(x - p1[0], y - p1[1])
        rc = math.hypot(x - p2[0], y - p2[1])
        return (x, y, max(ra, rb, rc))

    def _make_diameter(self, p0, p1):
        cx = (p0[0] + p1[0]) / 2.0
        cy = (p0[1] + p1[1]) / 2.0
        r0 = math.hypot(cx - p0[0], cy - p0[1])
        r1 = math.hypot(cx - p1[0], cy - p1[1])
        return (cx, cy, max(r0, r1))

    def is_in_circle(self, c, p):
        _MULTIPLICATIVE_EPSILON = 1 + 1e-14
        return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON

    def _cross_product(self, x0, y0, x1, y1, x2, y2):
        # Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
        return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


    def complexity(self):
        poly = self.projectxy()
        Ap = poly.area
        P = poly.length  # perim

        x, y = poly.buffer(0).exterior.xy

        circ = self._make_circle([x[i], y[i]] for i in range(len(x)))
        circle = Point(circ[0], circ[1]).buffer(circ[2])
        x, y = circle.exterior.xy
        Ac = circle.area

        # print(Ap, Ac, 0.1-(np.sqrt(Ac*Ap))
        self.cplx = 10 * (0.1 - (np.sqrt(Ac * Ap) / P ** 2))
        return (self.cplx)

