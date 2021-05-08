'''
plots aggregates from database using plotly
'''

from ipas.collection_from_db.calculations import ClusterCalculations
import plotly.graph_objs as go
import numpy as np


class PlotAgg:
    def __init__(self, agg, crystal_num):
        self.agg = agg
        self.points = agg.points
        self.x = self.points['x']
        self.y = self.points['y']
        self.z = self.points['z']
        self.crystal_num = crystal_num  # the crystal to plot


    def min_max_all_points(self):

        data = [self.x, self.y, self.z]
        concat_lists = [item.flatten() for item in data]
        concat_lists =np.hstack(concat_lists)
        self.min_all_points = min(concat_lists)
        self.max_all_points = max(concat_lists)

    def prism_points(self):

        prism_points = [] # hold prism points in order
        prismind = [0,6,7,1,2,8,9,3,4,10,11,5]  # prism line indices
        for n in prismind:
            prism_points.append(self.x[self.crystal_num][n])
            prism_points.append(self.y[self.crystal_num][n])
            prism_points.append(self.z[self.crystal_num][n])

        self.prism_points = prism_points


    def basal_points_top(self):

        basal_points_top = []
        for n in range(0,6): # basal face lines 
            basal_points_top.append(self.x[self.crystal_num][n])
            basal_points_top.append(self.y[self.crystal_num][n])
            basal_points_top.append(self.z[self.crystal_num][n])
        basal_points_top.append(self.x[self.crystal_num][0])
        basal_points_top.append(self.y[self.crystal_num][0])
        basal_points_top.append(self.z[self.crystal_num][0])

        self.basal_points_top = basal_points_top


    def basal_points_bottom(self):

        basal_points_bottom = []
        for n in range(6,12): # basal face lines 
            basal_points_bottom.append(self.x[self.crystal_num][n])
            basal_points_bottom.append(self.y[self.crystal_num][n])
            basal_points_bottom.append(self.z[self.crystal_num][n])
        basal_points_bottom.append(self.x[self.crystal_num][6])
        basal_points_bottom.append(self.y[self.crystal_num][6])
        basal_points_bottom.append(self.z[self.crystal_num][6])

        self.basal_points_bottom = basal_points_bottom


    def get_scatter_plot_data(self, line_width=6):
        '''
        since x y and z are all appended to the same
        array, get each coordinate separately indexing
        every third value starting at 0 for x, 1 for y, 
        and 2 for z
        '''

        x = self.prism_points[0::3]
        y = self.prism_points[1::3]
        z = self.prism_points[2::3]

        self.trace_prism = go.Scatter3d(x=x, y=y, z=z,
                                        mode='lines',
                                        line=dict(width = line_width,
                                                    color='black'),
                                        showlegend=False
                                       )

        x_basal_top = self.basal_points_top[0::3]
        y_basal_top = self.basal_points_top[1::3]
        z_basal_top = self.basal_points_top[2::3]

        self.trace_basal_top = go.Scatter3d(x=x_basal_top,
                                            y=y_basal_top,
                                            z=z_basal_top,
                                            mode='lines',
                                            line=dict(width=line_width,
                                                        color='black'),
                                            showlegend=False
                                           )

        x_basal_bottom = self.basal_points_bottom[0::3]
        y_basal_bottom = self.basal_points_bottom[1::3]
        z_basal_bottom = self.basal_points_bottom[2::3]

        self.trace_basal_bottom = go.Scatter3d(x=x_basal_bottom,
                                               y=y_basal_bottom,
                                               z=z_basal_bottom,
                                               mode='lines',
                                               line=dict(width=line_width,
                                                         color='black'),
                                               showlegend=False
                                              )


    def ellipsoid_surface(self):

        clus1 = ClusterCalculations(self.agg)
        A = clus1.fit_ellipsoid()
        rx, ry, rz = clus1.ellipsoid_axes_lengths(A)
        x, y, z = clus1.ellipsoid_axes_coords(rx, ry, rz)
        xell, yell, zell = clus1.ellipsoid_surface(A, x, y, z)
        
        maxXe = np.max(xell)
        minXe = np.min(xell)
        maxYe = np.max(yell)
        minYe = np.min(yell)
        maxZe = np.max(zell)
        minZe = np.min(zell)

        self.maxxyz = max(maxXe, maxYe, maxZe)
        self.minxyz = min(minXe, minYe, minZe)

        return xell, yell, zell, x, y, z


    def scatter_ellipsoid(self, xell, yell, zell, x, y, z, line_width=2):

        self.ellipsoid_surface = go.Scatter3d(x=xell[0::4].flatten(),
                                              y=yell[0::4].flatten(),
                                              z=zell[0::4].flatten(),
                                              mode='lines',
                                              line=dict(width=line_width,
                                                          color='black'),
                                              showlegend=False
                                             )
        self.ellipsoid_radius_b1 = go.Scatter3d(x=xell[0,:].flatten(),
                                              y=yell[0,:].flatten(),
                                              z=zell[0,:].flatten(),
                                              mode='lines',
                                              line=dict(width=6,
                                                        color='blue'),
                                              showlegend=False
                                             )
        self.ellipsoid_radius_b2 = go.Scatter3d(x=xell[20,:].flatten(),
                                              y=yell[20,:].flatten(),
                                              z=zell[20,:].flatten(),
                                              mode='lines',
                                              line=dict(width=6,
                                                        color='blue'),
                                              showlegend=False
                                             )
        self.ellipsoid_radius_r1 = go.Scatter3d(x=xell[10,:].flatten(),
                                              y=yell[10,:].flatten(),
                                              z=zell[10,:].flatten(),
                                              mode='lines',
                                              line=dict(width=6,
                                                        color='red'),
                                              showlegend=False
                                             )
        self.ellipsoid_radius_r2 = go.Scatter3d(x=xell[30,:].flatten(),
                                              y=yell[30,:].flatten(),
                                              z=zell[30,:].flatten(),
                                              mode='lines',
                                              line=dict(width=6,
                                                        color='red'),
                                              showlegend=False
                                             )
        self.ellipsoid_radius_g1 = go.Scatter3d(x=xell[:,20].flatten(),
                                              y=yell[:,20].flatten(),
                                              z=zell[:,20].flatten(),
                                              mode='lines',
                                              line=dict(width=6,
                                                        color='green'),
                                              showlegend=False
                                             )
        go.scatter3d.Textfont(size=30)


    def closest_pts(self, x_mesh_mono1, y_mesh_mono1, z_mesh_mono1,
                    x_mesh_mono2, y_mesh_mono2, z_mesh_mono2,
                    x_closest_mono, y_closest_mono, z_closest_mono,
                    x_closest_clus, y_closest_clus, z_closest_clus):
        
        # grid mesh over monomer1
        self.mesh_mono1 = go.Scatter3d(x=x_mesh_mono1,
                                        y=y_mesh_mono1,
                                        z=z_mesh_mono1,
                                        showlegend=False
                                       )
        # grid mesh over monomer1
        self.mesh_mono2 = go.Scatter3d(x=x_mesh_mono2,
                                        y=y_mesh_mono2,
                                        z=z_mesh_mono2,
                                        showlegend=False
                                       )
        # closest mesh grid pt on monomer1 
        # x value found after comparing closest grid pt to shapely returned nearest pt
        self.mesh_mono2 = go.Scatter3d(x=x_closest_clus,
                                        y=y_closest_clus,
                                        z=z_closest_clus,
                                        showlegend=False
                                       )
        # shapely closest pt on monomer2
        self.mesh_mono2 = go.Scatter3d(x=x_closest_mono,
                                        y=y_closest_mono,
                                        z=z_closest_mono,
                                        showlegend=False
                                       )
        
        
        
        
    def camera(self):

        self.camerax = dict(up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0.1, y=2.5, z=0.1)
                           )
        self.cameray = dict(up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=2.5, y=0.1, z=0.1)
                           )
        self.cameraz = dict(up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=0.1, y=0.1, z=2.5)
                           )


    def layout(self, tick_angle=20, showticklabels=True, tick_size=14,
                      tick_fontfamily='Times New Roman', showgrid=True):

        self.layout = go.Layout(scene = dict(camera=self.camerax,
                                            xaxis=dict(range=[self.minxyz, self.maxxyz],
                                                       title='X',
                                                       showgrid=showgrid,
                                                       tickangle=tick_angle,
                                                       showticklabels=showticklabels,
                                                       tickfont=dict(family=tick_fontfamily,
                                                                     size=tick_size)),
                                            yaxis=dict(range=[self.minxyz, self.maxxyz],
                                                       title='Y',
                                                       showgrid=showgrid,
                                                       tickangle=tick_angle,
                                                       showticklabels=showticklabels,
                                                       tickfont=dict(size=tick_size,
                                                                     family=tick_fontfamily)),
                                            zaxis=dict(range=[self.minxyz, self.maxxyz],
                                                       title='Z',
                                                       showgrid=showgrid,
                                                       tickangle=tick_angle,
                                                       showticklabels=showticklabels,
                                                       tickfont=dict(size=tick_size,
                                                                     family=tick_fontfamily))
                                            ),
                                font=dict(family='Sans',
                                          size=16)
                               )