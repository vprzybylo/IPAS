"""Sub class to IceCluster in ice_cluster_sql_master.py that holds methods to plot the aggregate(s)"""

import ipas
import math
import numpy as np
import shapely.geometry as geom
import shapely.affinity as sha
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
from descartes.patch import PolygonPatch
import descartes
from matplotlib.patches import Ellipse
import shapely.ops as shops
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import datetime
import operator

#Sub Class
class Plot_Cluster(ipas.Ice_Cluster):

    def __init__(self, cluster):
        # call parent constructor IceCluster
        super().__init__(cluster)

    def _crystal_projectxy(self, n):
        try:
            return geom.MultiPoint(self.points[n][['x', 'y']]).convex_hull
        except IndexError:
            return None

    def _crystal_projectxz(self, n):
        try:
            return geom.MultiPoint(self.points[n][['x', 'z']]).convex_hull
        except IndexError:
            return None

    def _crystal_projectyz(self, n):
        try:
            return geom.MultiPoint(self.points[n][['y', 'z']]).convex_hull
        except IndexError:
            return None
        
    def projectxy(self):
        polygons = [self._crystal_projectxy(n) for n in range(self.ncrystals) if self._crystal_projectxy(n) is not None]
        return shops.cascaded_union(polygons)

    def projectxz(self):
        polygons = [self._crystal_projectxz(n) for n in range(self.ncrystals) if self._crystal_projectxz(n) is not None]
        return shops.cascaded_union(polygons)

    def projectyz(self):
        polygons = [self._crystal_projectyz(n) for n in range(self.ncrystals) if self._crystal_projectyz(n) is not None]
        return shops.cascaded_union(polygons)
   
    def ellipse(self, u, v, rx, ry, rz):
        x = rx * np.cos(u) * np.cos(v)
        y = ry * np.sin(u) * np.cos(v)
        z = rz * np.sin(v)
        return x, y, z
        
        
    def _get_ellipsoid_points(self):
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
        return xell, yell, zell
    
    def _plot_crystal(self, ncrys, ax, color):  
        #plots individual monomers
        
        x = np.zeros(27)
        y = np.zeros(27)
        z = np.zeros(27)

        X = self.points['x'][ncrys]
        Y = self.points['y'][ncrys]
        Z = self.points['z'][ncrys]

        
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
#         ax.set_zticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.grid(False)

    def plot_ellipsoid_aggs(self, clusters, nearest_geoms_xz=None, nearest_geoms_yz=None, \
                            nearest_geoms_xy=None, view='x', circle=None, agg_agg=True):
        #plot multiple aggregates, each a different color
        xell, yell, zell = self._get_ellipsoid_points()

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        # 90, 0 for z orientation, 0, 90 for y orientation, 0, 0 for x orientation
        # ax.view_init(elev=90, azim=270)

        if view == 'x':
            ax.view_init(elev=0, azim=90)
        elif view == 'y':
            ax.view_init(elev=0, azim=0)
        elif view == 'z':
            ax.view_init(elev=90, azim=0)
        #else:
        #    ax.view_init(elev=0, azim=40)
        ax.plot_surface(xell, yell, zell, cstride=1, rstride=1, alpha=0.1)

        if agg_agg:
            start_list = [clus.ncrystals for clus in clusters]
            start = [0]+start_list
            end= [(np.sum([start[i], start[i+1]])+1) for i in range(len(start_list))] 
            colors=['k', 'r', 'b', 'darkgreen']
            for clus in range(len(clusters)): 
                #lowered color range so that darker colors are generated
                #color = list(np.random.choice(range(10), size=3)/10)
                color=colors[clus]
                for crys in range(start[clus], end[clus]-1):   
                    self._plot_crystal(crys, ax, color)
        else:
            start_list = [clus.ncrystals-1 for clus in clusters]
            start_list = [i+1 if i == 0 else i for i in start_list]
            start = [0]+start_list
            end= [np.sum([start[i], start[i+1]]) for i in range(len(start_list))] 
            colors=['k', 'r', 'b', 'darkgreen']
            for clus in range(len(clusters)): 
                #lowered color range so that darker colors are generated
                #color = list(np.random.choice(range(10), size=3)/10)
                color=colors[clus]
                for crys in range(start[clus], end[clus]):   
                    self._plot_crystal(crys, ax, color)
        if circle is not None:
            xcirc,ycirc = circle.exterior.xy
            ax.plot(xcirc,ycirc, color='green')
            maxXc = np.max(xcirc)
            minXc = np.min(xcirc)
            maxYc = np.max(ycirc)
            minYc = np.min(ycirc)
            maxXe = np.max(xell)
            minXe = np.min(xell)
            maxYe = np.max(yell)
            minYe = np.min(yell)
            maxZe = np.max(zell)
            minZe = np.min(zell)

            maxc = max(maxXc, maxYc, maxXe, maxYe, maxZe)
            minc = min(minXc, minYc, minXe, minYe, minZe)
            ax.set_xlim(minc, maxc)
            ax.set_ylim(minc, maxc)
            ax.set_zlim(minc, maxc)
        else:

            maxXe = np.max(xell)
            minXe = np.min(xell)
            maxYe = np.max(yell)
            minYe = np.min(yell)
            maxZe = np.max(zell)
            minZe = np.min(zell)

            maxxyz = max(maxXe, maxYe, maxZe)
            minxyz = min(minXe, minYe, minZe)

            ax.set_xlim(minxyz, maxxyz)
            ax.set_ylim(minxyz, maxxyz)
            ax.set_zlim(minxyz, maxxyz)

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


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

#         ax.set_zticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.grid(False)
#         plt.axis("off")

        # ax.view_init(30, i)
        # plt.pause(.001)

#        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#        fig.savefig('../plot_clusters/'+str(self.mono_phi)+'_'+current_time+'.png',rasterized=True, bbox_inches = 'tight', dpi=300, transparent=True)
        
        plt.show()
        #fig.clear()
        

    def ellipse_vertices_rotated(self, params, angle, leftverticex, rightverticex):
        # 2D
        # rotate axis points and reposition if off center

        leftverticey = params['xy'][1]
        rightverticey = params['xy'][1]

        xleft = ((leftverticex - params['xy'][0]) * np.cos(angle) - \
                    (leftverticey - params['xy'][1]) * np.sin(angle)) + params['xy'][0]

        xright = ((rightverticex - params['xy'][0]) * np.cos(angle) - \
                     (rightverticey - params['xy'][1]) * np.sin(angle)) + params['xy'][0]

        yleft = ((leftverticex - params['xy'][0]) * np.sin(angle) + \
                    (leftverticey - params['xy'][1]) * np.cos(angle)) + params['xy'][1]

        yright = ((rightverticex - params['xy'][0]) * np.sin(angle) + \
                     (rightverticey - params['xy'][1]) * np.cos(angle)) + params['xy'][1]

        x = [xleft, xright]
        y = [yleft, yright]

        return x, y


    def plot_axes_ellipse(self, params, ax):
        # 2D
        # major axis
        # orientation angle of ellipse in radians
        angle = params['angle'] * np.pi / 180
        leftverticex = params['xy'][0] - params['width'] / 2
        rightverticex = params['xy'][0] + params['width'] / 2
        x, y = self.ellipse_vertices_rotated(params,
                                             angle,
                                             leftverticex,
                                             rightverticex)
        ax.plot(x, y, color='k', linewidth=3)  # major/minor axis lines
        # minor axis
        leftverticex = params['xy'][0] - params['height'] / 2
        rightverticex = params['xy'][0] + params['height'] / 2
        angle = params['angle'] * np.pi / 180 + np.pi / 2
        x, y = self.ellipse_vertices_rotated(params,
                                             angle,
                                             leftverticex,
                                             rightverticex)
        ax.plot(x, y, color='k', linewidth=3)  # major/minor axis lines


    def plot_ellipse(self, dims, add_axes=True):

        # Only (x,z), (y,z), and (x,y) allowed for dimensions
        if dims == [['x', 'z']]:
            # self.rotate_to([np.pi / 2, 0, 0])
            poly = self.projectxz()
        elif dims == [['y', 'z']]:
            # self.rotate_to([np.pi / 2, np.pi / 2, 0])
            poly = self.projectyz()
        elif dims == [['x', 'y']]:
            # self.rotate_to([0, 0, 0])
            poly = self.projectxy()
        else:
            print('Not a valid dimension')

        params = self.fit_ellipse(dims)
        ellipse = Ellipse(**params)

        fig, ax = plt.subplots(1,1)
        ax.add_artist(ellipse)
        ellipse.set_alpha(.2)  # opacity
        ellipse.set_facecolor('darkorange')

        for l in range(len(dims)):

            crysmaxz = []
            crysminz = []
            maxzinds = []
            minzinds = []
            for i in range(self.ncrystals):
                hex1pts = self.points[dims[l]][i][0:6]  # first basal face
                poly1 = geom.Polygon([[p[0], p[1]] for p in hex1pts])  
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

                ax.plot(x1, y1, color=color, zorder=2, linewidth='2')  # edges of polygons
                ax.plot(x2, y2, color=color, zorder=4, linewidth='2')

        if add_axes:
            self.plot_axes_ellipse(params, ax)

        ax.set_aspect('equal', 'datalim')    
        plt.show()


