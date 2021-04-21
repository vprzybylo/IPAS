'''
Read database of IPAS aggregates 
(with and without point arrays)
'''

import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt


class Database:
    def __init__(self, files):
        self.files = files

    def read_database(self):
        '''
        read pickled files that hold aggregate and monomer attributes

        Params:
            - files (list): list of .pkl files to read (relative path)
        Returns:
            concatenated dataframe of all IPAS db files
        '''

        data = []
        for file in self.files:
            print('reading: ', file)
            data.append(pd.read_pickle(file, None))
        datapd = [pd.DataFrame(i) for i in data]
        self.df = pd.concat(datapd, axis=0, ignore_index=True)
        self.df = self.df.reset_index()


    def read_database_parquet(self):
        '''
        no points in parquet files
        '''
        self.df = dd.read_parquet(self.files, engine="pyarrow").compute()


    def shape(self, a, b, c):
        if (b-c) <= (a-b):
            return 'prolate'
        else:
            return 'oblate'


    def append_shape(self):
        '''
        appends a shape column to the dataframe as oblate or prolate spheroid
        based on a, b, and c of fit ellipsoid
        '''
        vfunc = np.vectorize(self.shape)
        self.df['shape'] = vfunc(self.df['a'], self.df['b'], self.df['c'])
        self.df = self.df.reset_index()
        self.df.loc[self.df['shape'] == 'oblate', 'agg_r'] = np.power((np.power(self.df['a'], 2) * self.df['c']), (1./3.))
        self.df.loc[self.df['shape'] == 'prolate', 'agg_r'] = np.power((np.power(self.df['c'], 2) * self.df['a']), (1./3.))


    def truncate_agg_r(self, limit):
        '''
        limit the agg size
        '''
        self.df = self.df[self.df.agg_r < limit]


    def append_agg_phi(self):
        '''
        appends a column for aggregate apsect ratio
        '''
        self.df['agg_phi'] = self.df.c/self.df.a


    def get_df_phi(self, phi_bins, i):
        #return a df that only queries within an aspect ratio bin
        return self.df[(self.df['agg_phi'] > phi_bins[i]) & (self.df['agg_phi'] < phi_bins[i+1])]


    def get_df_r(self, df_phi, r_bins, r):
        self.df_r = df_phi[(df_phi.agg_r > r_bins[r]) &  (df_phi.agg_r < r_bins[r+1])]


    def get_avg_ncrystals(self):
        return self.df_r['ncrystals'].mean()


    def get_avg_cplx(self):
        return self.df_r['cplx'].mean()


    def get_oblate_prolate_count(self):
        oblate_count = self.df_r['shape'][self.df_r['shape'] == 'oblate'].count()
        prolate_count = self.df_r['shape'][self.df_r['shape'] == 'prolate'].count()
        diff_count = oblate_count - prolate_count
        return (oblate_count, prolate_count, diff_count)


    def get_plate_columns_agg(self):
        agg_mono_plates = self.df_r['mono_phi'][self.df_r['mono_phi'] < 1.0].count()
        agg_mono_col = self.df_r['mono_phi'][self.df_r['mono_phi'] > 1.0].count()
        agg_mono_phi = (agg_mono_plates/(agg_mono_plates+agg_mono_col))*100
        return agg_mono_phi


    def get_avg_radius(self):
        return self.df_r['mono_r'].mean()


    def make_bar_plots(self, all_r_bins, variable, cmap, norm, phi_bin_labs,
                       x_axis_label, y_axis_label, cbar_format, cbar_label,
                       title, save_fig, save_name, agg_phi_bins=20,
                       agg_r_bins=20):

        fig, ax = plt.subplots(figsize=(10,7))

        for i in range(agg_phi_bins): 
            for r in range(agg_r_bins):
                if r != 0:
                    plt.bar([i]*20, all_r_bins[i,r+1], bottom= all_r_bins[i,r-1], 
                            color=cmap(norm(variable[i,r])), edgecolor='k')
                else:
                    plt.bar([i]*20, all_r_bins[i,r+1],
                            color=cmap(norm(variable[i,r])),
                            edgecolor='k')

        plt.yscale('log')
        plt.xticks(np.arange(len(phi_bin_labs)), phi_bin_labs, rotation=90, ha="center")
        plt.ylabel(y_axis_label)
        plt.xlabel(x_axis_label) 
        plt.title(title)
        cb = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(cb, format=cbar_format)
        cbar.ax.set_ylabel(cbar_label, fontsize=16, family='serif')

        plt.tight_layout()
        if save_fig:
            plt.savefig(save_name, format='png', dpi=300)
        plt.show()