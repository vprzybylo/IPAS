"""
Class representing ice clusters or aggregates.
Parent class that defines:
    - point arrays
    - crystals within the cluster
    - methods to move and reorient the arrays
"""

import copy as cp
import random

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
import shapely.ops as shops
from pyquaternion import Quaternion
from scipy import spatial
from shapely.geometry import Point
from shapely.ops import nearest_points

import ipas.collection_no_db.crystal as crys


def auto_str(cls):
    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%s" % item for item in vars(self).items()),
        )

    cls.__str__ = __str__
    return cls


@auto_str
class Cluster:
    """An aggregate"""

    def __init__(self, crystal, size=1):

        self.ncrystals = 1
        self.rotation = Quaternion()
        self.points = np.full(
            (1, 12), np.nan, dtype=[("x", float), ("y", float), ("z", float)]
        )
        self.points[0] = crystal.points
        self.add_points = None
        self.orient_points = None
        self.mono_phi = crystal.phi
        self.mono_r = crystal.r
        self.tol_ellipse = 10 ** -3

    def to_dict(self):
        return {
            "ncrystals": self.ncrystals,
            #              'xs':self.xs,
            #              'ys':self.ys,
            #              'zs':self.zs,
            "points": self.points,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "cplx": self.cplx,
            "phi2D": self.phi2D,
            "mono_phi": self.mono_phi,
            "mono_r": self.mono_r,
        }

    def add_cluster(self, cluster):
        self.points = np.vstack((self.points, cluster.points))
        self.ncrystals += cluster.ncrystals
        return self

    def remove_cluster(self, cluster):
        self.points = self.points[: -cluster.ncrystals]
        self.ncrystals -= cluster.ncrystals

    def add_crystal(self, crystal):
        self.points = np.vstack((self.points, crystal.points))
        self.ncrystals += crystal.ncrystals
        return self  # to make clus 3 instance

    def remove_crystal(self, crystal):
        self.points = self.points[: -crystal.ncrystals]
        self.ncrystals -= crystal.ncrystals

    def _crystals(self, i=None):
        # return a crystal with the same points and attributes as the
        # nth crystal in the cluster
        if i is None:
            crystals = []
            for n in range(self.ncrystals):
                cr = crys.Crystal(1, 1)
                cr.points = self.points[n]
                cr.rotation = self.rotation
                cx = cr.points["x"].mean()
                cy = cr.points["y"].mean()
                cz = cr.points["z"].mean()
                cr.center = [cx, cy, cz]
                cr.maxz = cr.points["z"].max()
                cr.minz = cr.points["z"].min()
                crystals.append(cr)
            return crystals
        else:
            cr = crys.Crystal(1, 1)
            cr.points = self.points[i]  # i = 0
            cr.rotation = self.rotation
            cx = cr.points["x"].mean()
            cy = cr.points["y"].mean()
            cz = cr.points["z"].mean()
            cr.center = [cx, cy, cz]
            cr.maxz = cr.points["z"].max()
            cr.minz = cr.points["z"].min()
            return cr

    def move(self, xyz):
        # move the entire cluster
        self.points["x"] += xyz[0]
        self.points["y"] += xyz[1]
        self.points["z"] += xyz[2]

    def _euler_to_mat(self, xyz):
        # Euler's rotation theorem, any rotation may be described using three angles
        [x, y, z] = xyz
        rx = np.matrix(
            [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
        )
        ry = np.matrix(
            [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
        )
        rz = np.matrix(
            [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
        )
        return rx * ry * rz

    def _rotate_mat(self, mat):
        points = cp.copy(self.points)
        self.points["x"] = (
            points["x"] * mat[0, 0] + points["y"] * mat[0, 1] + points["z"] * mat[0, 2]
        )
        self.points["y"] = (
            points["x"] * mat[1, 0] + points["y"] * mat[1, 1] + points["z"] * mat[1, 2]
        )
        self.points["z"] = (
            points["x"] * mat[2, 0] + points["y"] * mat[2, 1] + points["z"] * mat[2, 2]
        )

    def rotate_to(self, angles):
        # rotate to the orientation given by the 3 angles
        # get the rotation from the current position to the desired
        # rotation
        current_rot = self.rotation
        # print('current_rot', current_rot)
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
        x = np.mean(self.points[: self.ncrystals]["x"])
        y = np.mean(self.points[: self.ncrystals]["y"])
        z = np.mean(self.points[: self.ncrystals]["z"])
        return [x, y, z]

    def recenter(self):
        center_move = self.center_of_mass()
        self.move([-x for x in center_move])
        return center_move

    def _crystal_projectxy(self, n):
        return geom.MultiPoint(self.points[n][["x", "y"]])

    def _crystal_projectxz(self, n):
        return geom.MultiPoint(self.points[n][["x", "z"]])

    def _crystal_projectyz(self, n):
        return geom.MultiPoint(self.points[n][["y", "z"]])

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

    def evenly_spaced_mesh(self, nx=5):

        xi = np.linspace(np.min(self.points["x"]), np.max(self.points["x"]), nx)
        yi = np.linspace(np.min(self.points["y"]), np.max(self.points["y"]), nx)
        zi = np.linspace(np.min(self.points["z"]), np.max(self.points["z"]), nx)

        x, y, z = np.meshgrid(xi, yi, zi)
        return x, y, z

    def closest_point_mesh(self, shapely_pt, x, y, z):
        """
        find (x,y,z) on mesh grid that is closest to
        the closest points on the monomer returned from shapely
        """
        # use mesh as target points to search
        # transpose such that the array isnt all y then all z
        # but instead pairs of (y,z)
        grid_pts = np.array([y.ravel(), z.ravel()]).T
        tree = spatial.cKDTree(grid_pts)
        # grab k of the closest vertices to point returned from shapely
        # can be on any edge, vertex, or surface/face
        distance, index = tree.query([shapely_pt], k=1, n_jobs=-1)
        # now we can use the index to find x,y,z pt on mesh
        x_closest = x.ravel()[index[0]]
        y_closest = y.ravel()[index[0]]
        z_closest = z.ravel()[index[0]]

        return (x_closest, y_closest, z_closest)

    def combine(self, particle, plot_closest_pts=False):
        """
        aggregate two particles using the nearest points Shapely function
        particle can be either monomer (ice-agg) or aggregate (agg-agg)
        """

        # make sure cluster 2 is above cluster 1
        minclus2 = np.amin(particle.points["z"])
        maxclus1 = np.amax(self.points["z"])

        if minclus2 < maxclus1:
            diffmins = maxclus1 - minclus2
            particle.move([0, 0, diffmins])

        # find nearest points in xz, yz, and xy
        # NOT NECESSARILY ON VERTICES
        nearest_geoms_xz = nearest_points(self.projectxz(), particle.projectxz())
        nearest_geoms_yz = nearest_points(self.projectyz(), particle.projectyz())
        nearest_geoms_xy = nearest_points(self.projectxy(), particle.projectxy())

        # -------------------
        if plot_closest_pts:
            # print('BEFORE MOVING')
            self.plot_closest_pts(
                particle, nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy
            )
        # -------------------
        # first get closest points on cluster and monomer in y-z
        cluster_yz = np.array([nearest_geoms_yz[0].x, nearest_geoms_yz[0].y])
        particle_yz = np.array([nearest_geoms_yz[1].x, nearest_geoms_yz[1].y])

        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.view_init(elev=0, azim=0)

        # particle 1
        x, y, z = self.evenly_spaced_mesh()
        # ax.scatter(x, y, z, 'b', alpha=0.2)  # mesh
        (x_closest_clus, y_closest_clus, z_closest_clus) = self.closest_point_mesh(
            cluster_yz, x, y, z
        )
        # ax.scatter(x_closest_clus, y_closest_clus, z_closest_clus, color='r',s=100)
        # ax.scatter(0, cluster_yz[0], cluster_yz[1], color='g',s=100)

        # particle 2
        x, y, z = particle.evenly_spaced_mesh()
        # ax.scatter(x, y, z, alpha=0.2)  # mesh
        (
            x_closest_particle,
            y_closest_particle,
            z_closest_particle,
        ) = particle.closest_point_mesh(particle_yz, x, y, z)
        # ax.scatter(x_closest_particle, y_closest_particle, z_closest_particle, color='r',s=100)
        # ax.scatter(0, particle_yz[0], particle_yz[1], color='g',s=100)

        # particle.plot_crystal(ax, 'k')
        # self.plot_crystal(0, ax, 'k')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()

        #         if self.projectyz().contains(Point(y_closest_clus, z_closest_clus)):
        #             print('contained')

        #         else:
        #             print('not contained')

        # find the difference between closest points to move
        movex = x_closest_particle - x_closest_clus
        movey = y_closest_particle - y_closest_clus
        movez = z_closest_particle - z_closest_clus

        # print(movex, movey, movez)
        particle.move([-movex, -movey, -movez])

        # -------------------
        if plot_closest_pts:
            print("AFTER MOVING")
            self.plot_closest_pts(
                particle, nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy
            )

        return (nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy)

    def plot_closest_pts(
        self, cluster2, nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy
    ):

        self.add_crystal(cluster2)
        self.plot_ellipsoid_aggs(
            [self], nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy, view="x"
        )
        self.plot_ellipsoid_aggs(
            [self], nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy, view="y"
        )

        self.remove_crystal(cluster2)

    def add_crystal_from_above(self, crystal, lodge=0):
        # drop a new crystal onto the cluster
        # file = open('current_job.out', 'a+')
        # print('agg agg crystal from above',file=file)

        # use the bounding box to determine which crystals to get
        xmax = max(crystal.projectxy().exterior.coords.xy[0])
        ymax = max(crystal.projectxy().exterior.coords.xy[1])
        xmin = min(crystal.projectxy().exterior.coords.xy[0])
        ymin = min(crystal.projectxy().exterior.coords.xy[1])

        close = np.all(
            [
                self.points["x"][: self.ncrystals].max(axis=1) >= xmin,
                self.points["x"][: self.ncrystals].min(axis=1) <= xmax,
                self.points["y"][: self.ncrystals].max(axis=1) >= ymin,
                self.points["y"][: self.ncrystals].min(axis=1) <= ymax,
            ],
            axis=0,
        )

        which_close = np.where(close)

        close_crystals = [self._crystals(n) for n in which_close[0]]

        # see which crystals could actually intersect with the new crystal
        close_crystals = [
            x for x in close_crystals if x.projectxy().intersects(crystal.projectxy())
        ]
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
        mindiffz = (
            crystal.minz - first_hit_lower_bound
        )  # the largest it can possibly be

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
        rel_area = (
            self.projectxy().buffer(0).intersection(crystal2.projectxy().buffer(0))
        )
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
        c1_lines = [l for l in c1_bottom["lines"] if l.intersects(rel_area)]
        c2_lines = [l for l in c2_top["lines"] if l.intersects(rel_area)]
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
                        z1 = line1.interpolate(
                            (xy.x - line1.xy[0][0]) / (xrange1), normalized=True
                        ).z
                    else:
                        # interpolate using y value
                        z1 = line1.interpolate(
                            (xy.y - line1.xy[1][0]) / (line1.xy[1][1] - line1.xy[1][0]),
                            normalized=True,
                        ).z
                    if xrange2 != 0:
                        z2 = line2.interpolate(
                            (xy.x - line2.xy[0][0]) / (xrange2), normalized=True
                        ).z
                    else:
                        z2 = line2.interpolate(
                            (xy.y - line2.xy[1][0]) / (line2.xy[1][1] - line2.xy[1][0]),
                            normalized=True,
                        ).z
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz

        # 2) points and surfaces
        c1_points = [p for p in c1_bottom["points"] if p.intersects(rel_area)]
        c2_faces = [f for f in c2_top["faces"] if f.intersects(rel_area)]
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
                    z2 = (
                        -(point.x * normal_vec[0] + point.y * normal_vec[1] + d)
                        / normal_vec[2]
                    )
                    diffz = z1 - z2
                    if diffz < mindiffz:
                        mindiffz = diffz
                    # the point can only intersect one face, so we're
                    # done with this one
                    # break
                    # ^ I should be able to do that but I have to fix my 'bottom' function first!

        # 3) surfaces and points
        c1_faces = [f for f in c1_bottom["faces"] if f.intersects(rel_area)]
        c2_points = [p for p in c2_top["points"] if p.intersects(rel_area)]
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
                    z1 = (
                        -(point.x * normal_vec[0] + point.y * normal_vec[1] + d)
                        / normal_vec[2]
                    )
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

        """
        #Need to find lowest crystal that makes up the agg for intersection
        if self.ncrystals == 1:
            points = [ geom.Point(x) for x in self.points ]
            lowagg = min(points)
            if lowagg in points:
                print(True)
                self.points = self.points[x]
        else:
            points = self.points
        """

        lines = []
        faces = []

        p0 = self.points[0]
        p6 = self.points[6]
        if abs(p0["x"] - p6["x"]) < self.tol and abs(p0["y"] - p6["y"]) < self.tol:
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
                faces.append(geom.Polygon(list(self.points[i : (i + 6)])))
        elif abs(p0["z"] - p6["z"]) < self.tol:
            # lying flat on its side-- not returning hexagon faces
            if len(np.unique(self.points["z"])) == 4:
                # It's rotated so that there's a ridge on the top, and
                # the sides are vertical. Don't return any vertical
                # rectangular sides
                for n in range(5):
                    p1 = self.points[n]
                    p2 = self.points[n + 1]
                    # is it a non-vertical rectangle?
                    if (
                        abs(p1["x"] - p2["x"]) >= self.tol
                        and abs(p1["y"] - p2["y"]) >= self.tol
                    ):
                        faces.append(
                            geom.Polygon(
                                [
                                    self.points[n],
                                    self.points[n + 1],
                                    self.points[n + 7],
                                    self.points[n + 6],
                                ]
                            )
                        )
                # get that last rectangle missed
                p1 = self.points[5]
                p2 = self.points[0]
                if (
                    abs(p1["x"] - p2["x"]) >= self.tol
                    and abs(p1["y"] - p2["y"]) >= self.tol
                ):
                    faces.append(
                        geom.Polygon(
                            [
                                self.points[5],
                                self.points[0],
                                self.points[6],
                                self.points[11],
                            ]
                        )
                    )
                # get the lines around the hexagons
                for hexagon in range(2):
                    n0 = hexagon * 6
                    for i in range(5):
                        n = n0 + i
                        p1 = self.points[n]
                        p2 = self.points[n + 1]
                        if (
                            abs(p1["x"] - p2["x"]) >= self.tol
                            and abs(p1["y"] - p2["y"]) >= self.tol
                        ):
                            lines.append(
                                geom.LineString([self.points[n], self.points[n + 1]])
                            )
                    p1 = self.points[n0 + 5]
                    p2 = self.points[n0]
                    if (
                        abs(p1["x"] - p2["x"]) >= self.tol
                        and abs(p1["y"] - p2["y"]) >= self.tol
                    ):
                        lines.append(
                            geom.LineString([self.points[n0 + 5], self.points[n0]])
                        )
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
                faces.append(geom.Polygon(list(self.points[i : (i + 6)])))
            # get the rectangles
            for n in range(5):
                faces.append(
                    geom.Polygon(
                        [
                            self.points[n],
                            self.points[n + 1],
                            self.points[n + 7],
                            self.points[n + 6],
                        ]
                    )
                )
            # get that last rectangle I missed
            faces.append(
                geom.Polygon(
                    [self.points[5], self.points[0], self.points[6], self.points[11]]
                )
            )

        return {"lines": lines, "points": points, "faces": faces}

    def _top(self):
        self.bottom()

    def orient_cluster(self, rand_orient=False):
        """
        orient a cluster either randomly
        or to the rotation that maximizes
        the area in the xy plane
        """
        if rand_orient:  # random
            xrot = random.uniform(0, 2 * np.pi)
            yrot = random.uniform(0, 2 * np.pi)
            zrot = random.uniform(0, 2 * np.pi)
            self.rotate_to([xrot, yrot, 0])

        else:  # flat
            area_og = 0
            for i in np.arange(0.0, np.pi / 2, 0.01):
                self.rotate_to([i, 0, 0])
                area = self.projectxy().area
                if area > area_og:
                    xrot = i
                    area_og = area
                self.points = self.add_points

            area_og = 0
            for i in np.arange(0.0, np.pi / 2, 0.01):
                self.rotate_to([0, i, 0])
                area = self.projectxy().area

                if area > area_og:
                    yrot = i
                    area_og = area
                self.points = self.add_points

            zrot = random.uniform(0, np.pi / 2)

            best_rot = [xrot, yrot, zrot]
            self.rotate_to(best_rot)
