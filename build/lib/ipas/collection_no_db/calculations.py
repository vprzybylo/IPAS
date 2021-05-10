'''
Finds fit ellipse and ellipsoids surrounding clusters
Performs any external calculations on the clusters 
such as aspect ratio, complexity, etc.
'''

import ipas.collection_no_db.plot as plot
import ipas.collection_no_db.cluster as clus
import numpy.linalg as la
import math
import numpy as np
import shapely.geometry as geom
import shapely.affinity as sha
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
import shapely.ops as shops


#subclass
class ClusterCalculations(plot.PlotCluster, clus.Cluster):

    def __init__(self, cluster):
        # call parent constructor 
        super().__init__(cluster) 


    def fit_ellipsoid(self, tol=0.01):  # mve = minimum volume ellipse
        # Based on work by Nima Moshtagh
        # http://www.mathworks.com/matlabcentral/fileexchange/9542

        """
        Finds the minimum volume enclosing ellipsoid (3D)
        of a set of data in "center form"
        (x-c).T * A * (x-c) = 1

        Outputs:
        -------
        centroid (c) : D-dimensional vector containing
                        the center of the ellipsoid.
        A : This matrix contains all the information
            regarding the shape of the ellipsoid.

        To get the radii and orientation of the ellipsoid,
        take the Singular Value Decomposition
        of the output matrix A:

        [U Q V] = svd(A);

        the radii are given by:
        r1 = 1/sqrt(Q(1,1));
        r2 = 1/sqrt(Q(2,2));
        ...
        rD = 1/sqrt(Q(D,D));

        and matrix V is the rotation matrix that gives
        the orientation of the ellipsoid.
        """

        # only run on vertices of hexagonal prisms
        points_arr = np.concatenate(self.points)[:self.ncrystals * 12]
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

        self.centroid = np.dot(u, points_arr)

        A = la.inv(np.dot(np.dot(points_arr.T, np.diag(u)), points_arr) -
                   np.multiply.outer(self.centroid, self.centroid)) / d

        return A


    def ellipsoid_axes_lengths(self, A):

        U, D, V = la.svd(A)  # singular-value decomposition
        # D is a diagonal matrix
        rx, ry, rz = 1. / np.sqrt(D)  
        self.a, self.b, self.c = \
                sorted([rx, ry, rz], reverse=True)
        return rx, ry, rz


    def ellipsoid_axes_coords(self, rx, ry, rz):
        
        # Cartesian coordinates that correspond
        # to the spherical angles:
        u, v = np.mgrid[0:2 * np.pi:40j, -np.pi / 2:np.pi / 2:40j]
        x = rx * np.cos(u) * np.cos(v)
        y = ry * np.sin(u) * np.cos(v)
        z = rz * np.sin(v)
        return x, y, z


    def ellipsoid_surface(self, A, x, y, z):

        # singular-value decomposition
        _, _, V = la.svd(A)
        E = np.dstack([x, y, z])
        self.E = np.dot(E, V) + self.centroid
        
        xell, yell, zell = np.rollaxis(self.E, axis=-1)

        return xell, yell, zell


    def fit_ellipse(self, dims):
        '''
        2D ellipse
        Emulating this function, but for polygons in continuous
        space rather than blobs in discrete space:
        http://www.idlcoyote.com/ip_tips/fit_ellipse.html
        '''

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

        return ellipse
    

    def _get_moments(self, poly):
        '''
        get 'mass moments' for this cluster's 2D polygon using a
        variation of the shoelace algorithm
        '''
        
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

        if np.isnan(phi).any():
            self.plot_ellipse(dims=[['y', 'z']])
            self.plot_ellipse(dims=[['x', 'z']])
            self.plot_ellipse(dims=[['x', 'y']])

        self.phi2D = sum(phi) / len(phi)

        return self.phi2D


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
        '''
        Calculate particle complexity
        from Schmitt (2010)
        '''
        poly3 = self.projectxy()
        Ap = poly3.area
        P = poly3.length
    
        #Ap = poly1.area+poly2.area
        
        #in the case that the clusters don't perfectly touch, we are still summing both cluster perims
        #since there is such minimal overlap between clusters
        #P = poly1.length +poly2.length # perim
        
        #next line is for a convex hull around both clusters (even if they are not touching)
        
        try:
            multipt = [geom.MultiPoint(self.points[n][['x','y']]) for n in range(self.ncrystals)]
        except IndexError:
            print('in index error in cplx')
            multipt = None
            
        if multipt is not None:
            poly = shops.cascaded_union(multipt).convex_hull
            x, y = poly.exterior.xy

            circ = self._make_circle([x[i], y[i]] for i in range(len(x)))
            circle = Point(circ[0], circ[1]).buffer(circ[2])
            x, y = circle.exterior.xy
            Ac = circle.area
            
            self.cplx = 10 * (0.1 - (np.sqrt(Ac * Ap) / P ** 2))
            #print('Ap, Ac, P cplx= ', Ap, Ac, P, self.cplx)
            return (self.cplx, circle)
        else:
            return -999, None