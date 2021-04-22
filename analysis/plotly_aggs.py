'''
plots aggregates from database using plotly
'''

import plotly.graph_objs as go
import numpy as np

class PlotAgg:
    def __init__(self, points, crystal_num):
        self.points = points
        self.x = points['x']
        self.y = points['y']
        self.z = points['z']
        self.crystal_num = crystal_num  # the crystal to plot


    def min_all_points(self):
        return np.min([self.x, self.y, self.z])

    def max_all_points(self):
        return np.max([self.x, self.y, self.z])

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


    def get_xyz_for_tracing(self):
        '''
        since x y and z are all appended to the same
        array, get each coordinate separately indexing
        every third value starting at 0 for x, 1 for y, 
        and 2 for z
        '''

        self.x = self.prism_points[0::3]
        self.y = self.prism_points[1::3]
        self.z = self.prism_points[2::3]

        self.x_basal_top = self.basal_points_top[0::3]
        self.y_basal_top = self.basal_points_top[1::3]
        self.z_basal_top = self.basal_points_top[2::3]

        self.x_basal_bottom = self.basal_points_bottom[0::3]
        self.y_basal_bottom = self.basal_points_bottom[1::3]
        self.z_basal_bottom = self.basal_points_bottom[2::3]


    def scatter(self):
        trace_prism = go.Scatter3d(x=self.x, y=self.y, z=self.z,
                             mode = 'lines',
                             line = dict(width = 8,
                                         color='#29569F'),
                             showlegend=False)
        trace_basal_top = go.Scatter3d(x=self.x_basal_top,
                                  y=self.y_basal_top,
                                  z=self.z_basal_top,
                                  mode = 'lines',
                                  line = dict(width = 8,
                                              color='#52AAFC'),
                                  showlegend=False)
        trace_basal_bottom = go.Scatter3d(x=self.x_basal_bottom,
                                  y=self.y_basal_bottom,
                                  z=self.z_basal_bottom,
                                  mode = 'lines',
                                  line = dict(width = 8, color='#4B0089'),
                                  showlegend=False)
        self.plotting_data = [trace_prism, trace_basal_top, trace_basal_bottom]



    def set_up_camera(self):
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


    def set_up_layout(self, tick_angle=20, width=800, height=800,
                      showticklabels=True, tick_size=14,
                      tick_fontfamily='Times New Roman', showgrid=True):

        self.layout = go.Layout(scene = dict(camera=self.camerax,
                                        xaxis=dict(title='X',
                                                   showgrid=showgrid,
                                                   tickangle = tick_angle,
                                                   showticklabels=showticklabels,
                                                   range=[self.min_all_points, self.max_all_points],
                                                   tickfont=dict(family=tick_fontfamily,
                                                                 size=14)),
                                        yaxis=dict(title='Y',
                                                   showgrid=showgrid,
                                                   tickangle = tick_angle,
                                                   showticklabels=showticklabels,
                                                   range=[self.min_all_points, self.max_all_points],
                                                   tickfont=dict(size=14,
                                                                 family=tick_fontfamily)),
                                        zaxis=dict(title='Z',
                                                   showgrid=showgrid,
                                                   tickangle = tick_angle,
                                                   nticks=5,
                                                   showticklabels=showticklabels,
                                                   range=[self.min_all_points, self.max_all_points],
                                                   tickwidth=1,
                                                   tickfont=dict(size=14,
                                                                 family=tick_fontfamily))),
                                font=dict(family='Courier New',
                                          size=24),
                                autosize=False,
                                width=width,
                                height=height
                                )