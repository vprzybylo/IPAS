"""
Class representing ice crystals (monomers)
"""

import copy as cp
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import shapely.geometry as geom
from pyquaternion import Quaternion
from scipy import spatial
from shapely.geometry import Point
from shapely.ops import nearest_points


def auto_str(cls):
    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%s" % item for item in vars(self).items()),
        )

    cls.__str__ = __str__
    return cls


@auto_str
class Crystal:
    """A hexagonal prism representing a single ice crystal."""

    def __init__(self, a, c, center=[0, 0, 0], rotation=[0, 0, 0]):
        self.a = a
        self.c = c
        self.phi = self.c / self.a
        self.r = int(np.round(np.power((np.power(self.a, 2) * self.c), (1.0 / 3.0))))
        self.center = [0, 0, 0]  # start the crystal at the origin
        self.rotation = Quaternion()
        self.ncrystals = 1
        self.hold_clus = None
        self.tol = 10 ** -11  # used for some calculations

        # put together the hexagonal prism
        ca = c * 2  # diameter
        mf = a * 2  # diameter
        f = np.sqrt(3) / 4  # convenient number for hexagons
        x1 = ca / 2

        # creates 12 point arrays for hexagonal prisms

        if c < a:  # initialize plates so that the basal face is falling down
            self.points = np.array(
                [
                    (mf * f, -mf / 4, x1),
                    (mf * f, mf / 4, x1),
                    (0, mf / 2, x1),
                    (-mf * f, mf / 4, x1),
                    (-mf * f, -mf / 4, x1),
                    (0, -mf / 2, x1),
                    (mf * f, -mf / 4, -x1),
                    (mf * f, mf / 4, -x1),
                    (0, mf / 2, -x1),
                    (-mf * f, mf / 4, -x1),
                    (-mf * f, -mf / 4, -x1),
                    (0, -mf / 2, -x1),
                ],
                dtype=[("x", float), ("y", float), ("z", float)],
            )

        else:  # initialize points so that columns fall prism face down
            self.points = np.array(
                [
                    (x1, -mf / 4, mf * f),
                    (x1, mf / 4, mf * f),
                    (x1, mf / 2, 0),
                    (x1, mf / 4, -mf * f),
                    (x1, -mf / 4, -mf * f),
                    (x1, -mf / 2, 0),
                    (-x1, -mf / 4, mf * f),
                    (-x1, mf / 4, mf * f),
                    (-x1, mf / 2, 0),
                    (-x1, mf / 4, -mf * f),
                    (-x1, -mf / 4, -mf * f),
                    (-x1, -mf / 2, 0),
                ],
                dtype=[("x", float), ("y", float), ("z", float)],
            )

        self.rotate_to(rotation)  # rotate the crystal
        self.move(center)  # move the crystal to center
        self.maxz = self.points["z"].max()
        self.minz = self.points["z"].min()

    def move(
        self, xyz
    ):  # moves the falling crystal anywhere over the seed crystal/aggregate within the max bounds
        self.points["x"] += xyz[0]
        self.points["y"] += xyz[1]
        self.points["z"] += xyz[2]
        # update the crystal's center:
        for n in range(3):
            self.center[n] += xyz[n]

    def _center_of_mass(self):
        x = np.mean(self.points["x"])
        y = np.mean(self.points["y"])
        z = np.mean(self.points["z"])
        return [x, y, z]

    def recenter(self):
        self.move([-x for x in self._center_of_mass()])

    def add_crystal(self, crystal):
        self.points = np.vstack((self.points, crystal.points))
        self.ncrystals += crystal.ncrystals
        return self  # to make clus 3 instance

    def remove_crystal(self, crystal):
        self.points = self.points[: -crystal.ncrystals]
        self.ncrystals -= crystal.ncrystals

    def remove_cluster(self, crystal):
        self.points = self.points[: -crystal.ncrystals]
        self.ncrystals -= crystal.ncrystals

    def _rotate_mat(self, mat):  # when a crystal is rotated, rotate the matrix with it
        points = cp.deepcopy(self.points)
        self.points["x"] = (
            points["x"] * mat[0, 0] + points["y"] * mat[0, 1] + points["z"] * mat[0, 2]
        )
        self.points["y"] = (
            points["x"] * mat[1, 0] + points["y"] * mat[1, 1] + points["z"] * mat[1, 2]
        )
        self.points["z"] = (
            points["x"] * mat[2, 0] + points["y"] * mat[2, 1] + points["z"] * mat[2, 2]
        )

    def _euler_to_mat(self, xyz):
        # Euler's rotation theorem, any rotation may be described using three angles.
        # takes angles and rotates coordinate system
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

    def rotate_to(self, angles):

        # rotate to the orientation given by the 3 angles
        # get the rotation from the current position to the desired rotation

        rmat = self._euler_to_mat(angles)
        desired_rot = Quaternion(matrix=rmat)

        rot_mat = (desired_rot * self.rotation.inverse).rotation_matrix
        self._rotate_mat(rot_mat)

        # update the crystal's center:
        xyz = ["x", "y", "z"]
        for n in range(3):
            self.center[n] = self.points[xyz[n]].mean()
        self.rotation = desired_rot
        return self

    def orient_crystal(self, rand_orient=False):
        """
        orient a crystal either randomly or to the
        rotation that maximizes the area in the x-y plane
        """

        if rand_orient:
            xrot = random.uniform(0, 2 * np.pi)
            yrot = random.uniform(0, 2 * np.pi)
            zrot = random.uniform(0, 2 * np.pi)
            self.rotate_to([xrot, yrot, zrot])

        else:

            area_og = 0
            for i in np.arange(0.0, np.pi / 2, 0.1):
                self.rotate_to([i, 0, 0])
                area = self.projectxy().area
                if area > area_og:
                    xrot = i
                    area_og = area
                self.points = self.hold_clus

            area_og = 0
            for i in np.arange(0.0, np.pi / 2, 0.1):
                self.rotate_to([0, i, 0])
                area = self.projectxy().area
                if area > area_og:
                    yrot = i
                    area_og = area
                self.points = self.hold_clus

            zrot = random.uniform(0, 2 * np.pi)
            best_rot = [xrot, yrot, zrot]
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
            # point doesn't need to hit polygon for agg-agg since moving to closest
            # anyways
            list_of_points.append(pnt)

        agg_pt = list_of_points[0]
        new_pt = list_of_points[1]
        return (agg_pt, new_pt)

    def closest_points(self, cluster2):

        minclus2 = np.amin(cluster2.points["z"])
        maxclus1 = np.amax(self.points["z"])

        if minclus2 < maxclus1:
            diffmins = maxclus1 - minclus2
            # self.plot_ellipsoid()
            # cluster2.plot_ellipsoid()
            # print('moving cluster2 up')
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

        # print('movez_xz, movez_yz', movez_xz, movez_yz)
        cluster2.move([-movex, -movey, -(max(abs(movez_xz), abs(movez_yz)))])
        # print('pts1', cluster2.points['x'].max(), cluster2.points['y'].max())
        # move in x-y
        movex = nearest_geoms_xy[1].x - nearest_geoms_xy[0].x
        movey = nearest_geoms_xy[1].y - nearest_geoms_xy[0].y
        # if movex != 0.0 or movey != 0.0:
        #    print('moving x-y', movex, movey)

        cluster2.move([-movex, -movey, 0])

        return (nearest_geoms_xz, nearest_geoms_yz, nearest_geoms_xy)

    def evenly_spaced_mesh(self, nx=10):

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

    def projectxy(self):
        return geom.MultiPoint(self.points[["x", "y"]]).convex_hull

    def projectxz(self):
        return geom.MultiPoint(self.points[["x", "z"]]).convex_hull

    def projectyz(self):
        return geom.MultiPoint(self.points[["y", "z"]]).convex_hull

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

        A = (
            la.inv(
                np.dot(np.dot(points_arr.T, np.diag(u)), points_arr)
                - np.multiply.outer(c, c)
            )
            / d
        )
        return A, c

    def ellipsoid_axes(self):
        """
        semi-principal axes of fit-ellipsoid
        in descending order
        """
        A, c = self._mvee()
        U, D, V = la.svd(A)  # singular-value decomposition
        rx, ry, rz = 1.0 / np.sqrt(D)  # D is a diagonal matrix
        self.agg_a, self.agg_b, self.agg_c = sorted([rx, ry, rz], reverse=True)
        return self.agg_a, self.agg_b, self.agg_c

    def plot_crystal(self, ax, color):
        # plots individual monomers

        x = np.zeros(27)
        y = np.zeros(27)
        z = np.zeros(27)

        X = self.points["x"]
        Y = self.points["y"]
        Z = self.points["z"]

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

    def _bottom(self):
        # return geometry of bottom side of falling crystal
        # to be used in connecting bottom of one crystal to the top of the other
        # getting the same points regardless of the orientation
        points = [geom.Point(list(x)) for x in self.points]
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

        # return the geometry representing the bottom side of the prism

        # # similar to projectxy
        # if self.rotation[1] == math.pi / 2:
        #     # it's vertical, so just return one of the hexagons
        #     points = self.points[0:6]

        # first find top and bottom hexagon

        # remove the top two points

        # make the lines

        # make the faces

        return {"lines": lines, "points": points, "faces": faces}

    def _top(self):
        # return the geometry representing the top side of the prism

        # first find top and bottom hexagon

        # remove the bottom two points

        # make the lines

        # make the faces

        # return {'lines': lines, 'points': points, 'faces': faces}

        # temporary, until I fix these functions
        top = self._bottom()
        # # unless it's vertical
        # if self.rotation[1] / (np.pi / 2) % 4 == 1:
        #     top['points'] = [ geom.Point(list(x)) for x in self.points[0:6] ]
        #     top['lines'] = []
        #     for i in range(5): # get the points around each hexagon
        #         top['lines'].append(geom.LineString([self.points[i], self.points[i + 1]]))
        #     top['lines'].append(geom.LineString([self.points[5], self.points[0]]))
        # elif self.rotation[1] / (np.pi / 2) % 4 == 3:
        #     top['points'] = [ geom.Point(list(x)) for x in self.points[6:12] ]
        #     top['lines'] = []
        #     for i in range(5): # get the points around each hexagon
        #         top['lines'].append(geom.LineString([self.points[i + 6], self.points[i + 7]]))
        #         top['lines'].append(geom.LineString([self.points[11], self.points[6]]))

        return top

    def _min_vert_dist(self, crystal2):
        # find the minimum directed distance to crystal2 traveling straight downward

        rel_area = (
            self.projectxy().buffer(0).intersection(crystal2.projectxy().buffer(0))
        )

        if not isinstance(rel_area, geom.Polygon):
            print("bad poly", file=current_job.out)
            print(rel_area)

            return None
        c1_bottom = self._bottom()
        c2_top = crystal2._top()
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
