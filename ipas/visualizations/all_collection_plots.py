'''
plotting functions that are called in all_collection_plots.ipynb
'''

import scipy.stats as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def change_in_density_nm(dds_Ntot_rand, dds_Ntot_flat, savefig=False):

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(11,5))

    cmap = plt.cm.jet
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, 8, 9)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    #phiarr = [0.01, 0.1, 1.0, 10.0, 100.0]
    phiarr = [0.01, 0.1, 0.5, 1.0, 2.0, 10., 100.]
    Ns = np.arange(9)
    for N in Ns:
        ax1.scatter(phiarr, np.mean(dds_Ntot_rand[:,0,:,N], axis=1),
                    c=cmap(norm(N),8), s=200)
    ax1.set_xlim(0.005, 150)
    #ax1.set_ylim(-0.15, 0.0)
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    ax1.set_xscale('log')
    ax1.set_ylabel('Change in Density')
    ax1.set_title('Random Orientation')
    ax1.set_xlabel('Monomer Aspect Ratio')
    cb = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.92,0.1,0.02,0.8])  #[left, bottom, width, height] 
    cbar = plt.colorbar(cb, ticks=np.linspace(0, 9, 10) + .5, format='%d', cax=cax)
    cbar.set_ticklabels([2,3,4,5,6,7,8,9,10]);
    cbar.set_label('Number of Monomers');
    ax1.set_xticks(phiarr)

    for N in Ns:
        ax2.scatter(phiarr, np.mean(dds_Ntot_flat[:,0,:,N], axis=1),
                    c=cmap(norm(N),8), s=200)
    ax2.set_xlim(0.005, 200)
    #ax2.set_ylim(-0.15, 0.0)
    ax2.set_xscale('log')
    ax2.set_title('Quasi-Horizontal Orientation')
    ax2.set_xlabel('Monomer Aspect Ratio')
    ax2.set_xticks(phiarr);
    if savefig:
        plt.savefig('../plots/dd_nmono_bothorientations.pdf',
                    format='pdf', dpi=300, bbox_inches='tight')


def change_in_density_aggr(agg_as_Ntot_rand, agg_as_Ntot_flat,
                             agg_cs_Ntot_rand, agg_cs_Ntot_flat,
                             dds_Ntot_rand, dds_Ntot_flat):

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10,5))
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(-15, 0.0, 10)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    n_perc = dds_Ntot_rand*100
    phiarr = [0.01, 0.1, 0.5, 1.0, 2.0, 10.,  100.]
    Ns = np.arange(9)
    for N in Ns:
        agg_r = np.round(np.power((np.power(np.mean(agg_as_Ntot_rand[:,0,:,N], axis=1),2)
                                   *np.mean(agg_cs_Ntot_rand[:,0,:,N], axis=1)),(1./3.)))
        ax1.scatter(phiarr, agg_r, c=cmap(norm(np.mean(n_perc[:,0,:,N], axis=1))), s=N*50)
    ax1.set_xlim(0.07, 15)
    ax1.set_xscale('log')
    ax1.set_ylabel('Avgerage Aggregate Radius [\u03BCm]')
    ax1.set_xlabel('Monomer Aspect Ratio')
    ax1.set_title('Random Orientation')
    cb = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.92,0.1,0.02,0.8])  #[left, bottom, width, height] 
    cbar = plt.colorbar(cb, cax=cax)
    cbar.set_label('Change in Density [%]');

    dds_Ntot_flat_perc = dds_Ntot_flat*100
    for N in Ns:
        agg_r = np.round(np.power((np.power(np.mean(agg_as_Ntot_flat[:,0,:,N], axis=1),2)
                                   *np.mean(agg_cs_Ntot_flat[:,0,:,N], axis=1)),(1./3.)))
        ax2.scatter(phiarr, agg_r, c=cmap(norm(np.mean(dds_Ntot_flat_perc[:,0,:,N], axis=1))), s=N*50)
    ax2.set_xlim(0.07, 15)
    ax2.set_xscale('log')
    ax2.set_xlabel('Monomer Aspect Ratio')
    ax2.set_title('Quasi-Horizontal Orientation')


def dd_rand_hist(dds_iceice_rand, dds_iceagg_rand, dds_aggagg_rand, savefig=False):

    binwidth = 0.1
    mindata = min(np.amin(dds_iceice_rand), np.amin(dds_iceagg_rand), np.amin(dds_aggagg_rand))
    maxdata = max(np.amax(dds_iceice_rand), np.amax(dds_iceagg_rand), np.amax(dds_aggagg_rand))
    bins=np.arange(mindata, maxdata + binwidth, binwidth)
    _, n,bins = plt.hist(dds_aggagg_rand.ravel(), color='blue', alpha=0.5, bins=100, label='agg-agg', density=True)
    _, n,bins = plt.hist(dds_iceagg_rand.ravel(), color='orange', alpha=0.5, bins=100, label='ice-agg', density=True)
    _, n,bins = plt.hist(dds_iceice_rand.ravel(), color='green', alpha=0.5, bins=100, label='ice-ice', density=True)

    plt.gca().set(xlabel='Density Change', ylabel='Frequency');
    plt.xlim(-1., 0.8)
    plt.legend()
    if savefig:
        plt.savefig('../plots/dd_rand_hist.png')


def adjacent_values(vals, q1, q3):

    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):

    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_xlabel('Collection Method', fontsize=16)
    ax.set_title('Quasi-Horizontal Orientation', fontsize=16)


def violin_density_plots(dds_iceice_flat, dds_iceagg_flat, dds_aggagg_flat):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    colors=['g','b','r']
    dataset = [dds_iceice_flat.ravel(), dds_iceagg_flat.ravel(), dds_aggagg_flat.ravel()]
    for c, data in enumerate(dataset): 
        data= data[(data<np.quantile(data, .95)) & (data>np.quantile(data, .05))]
        if c == 2:
            points = 1000
        else:
            points = 100
        parts = ax.violinplot(data,showmeans=False, showmedians=False,
                showextrema=False, positions=[c], widths =0.8)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[c])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        quartile1, median, quartile3 = np.percentile(data, [25, 50, 75])
        ax.scatter(c, median, marker='o', color='white', s=30, zorder=3)
        whiskers = np.array([adjacent_values(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
        ax.vlines(c, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(c, whiskersMin, whiskersMax, color='k', linestyle='-', lw=5)

    plt.axhline(y=0.0, color='k', linestyle='--', alpha = 0.5)
    ax.set_ylabel('Change in Density', fontsize=16)
    #ax.set_ylim(-0.6, 0.5)
    # set style for the axes
    labels = ['ice-ice', 'ice-agg', 'agg-agg']
    set_axis_style(ax, labels)
    plt.show()


def phi_pdfs_ba_ca_rand(agg_as_iceagg_rand, agg_bs_iceagg_rand, agg_cs_iceagg_rand, 
                        phi_ba_iceagg_rand, phi_ca_iceagg_rand,
                        phi_ba_aggagg_rand, phi_ca_aggagg_rand, savefig=False):

    fig, axes = plt.subplots(2,5, figsize=(16,5), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    axs = axes.ravel()

    phi_bins_rand=[0.00548, 0.273, 0.327, 0.359, 0.384, 0.405,
     0.424, 0.441, 0.458, 0.474, 0.490, 0.506,
     0.522, 0.539, 0.557, 0.576, 0.598, 0.623,
     0.655, 0.701, 0.975]

    print('phi \t oblates\t prolates \tprolates majority?')
    r=9
    i=0
    phios = [0,3,5,7,9,11,13,15,17,19]
    for phio in phios:
        iceagg = np.vstack([phi_ba_iceagg_rand[phio,r,:], phi_ca_iceagg_rand[phio,r,:]])
        kde_iceagg = st.gaussian_kde(iceagg)    
        aggagg = np.vstack([phi_ba_aggagg_rand[phio,r,:], phi_ca_aggagg_rand[phio,r,:]])
        kde_aggagg = st.gaussian_kde(aggagg)

        oblates = 0
        prolates = 0
        for l in range(300):
            if (agg_bs_iceagg_rand[phio,r,l] - agg_cs_iceagg_rand[phio,r,l]) <= \
                (agg_as_iceagg_rand[phio,r,l] - agg_bs_iceagg_rand[phio,r,l]):
                prolates +=1
            else:
                oblates+=1
        if prolates>oblates:
            bigger = 'True'
        else:
            bigger = ' '

        print('%3.1f, %10.2f, %15.2f, %15s' %(phio, (oblates/300)*100, (prolates/300)*100, bigger))

        # evaluate on a regular grid
        xgrid = np.arange(0.0, 1.01, 0.01)
        ygrid = np.arange(0.0, 1.01, 0.01)

        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z_iceagg = kde_iceagg.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_iceagg = Z_iceagg.reshape(Xgrid.shape)
        Z_aggagg = kde_aggagg.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_aggagg = Z_aggagg.reshape(Xgrid.shape)

        axs[i].contour(Z_iceagg, cmap='Blues', vmin=1, vmax=10)
        axs[i].contour(Z_aggagg, cmap='Reds', vmin=1, vmax=10)

        #modes iceagg
        modes = np.where(Z_iceagg==np.max(Z_iceagg))
        #print(float(modes[1]/100), float(modes[0]/100))
        axs[i].plot(float(modes[1]), float(modes[0]), 'bo')

        #modes aggagg
        modes = np.where(Z_aggagg==np.max(Z_aggagg))
        #print(float(modes[1]/100), float(modes[0]/100))


        axs[i].plot(float(modes[1]), float(modes[0]), 'ro')

        axs[i].plot(np.linspace(0.0,100,10),np.linspace(0.0,100,10), 'gray', '--', zorder=2)
        axs[i].set_xticks([0,20,40,60,80,100])
        axs[i].set_yticks([0,20,40,60,80,100])
        #axs[i].set_xticklabels([])
        axs[i].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=50)
        axs[i].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[i].grid(which='major', alpha=0.5)

        axs[i].set_title('[%.3f-%.3f]' %(phi_bins_flat[phio], phi_bins_flat[phio+1]), fontfamily='serif')

        i+=1
    m = plt.cm.ScalarMappable(cmap='Reds')
    m.set_array(Z_aggagg)

    c = plt.cm.ScalarMappable(cmap='Blues')
    c.set_array(Z_iceagg)

    cbar=plt.colorbar(m, boundaries=np.linspace(0, 50, 11),ax=axes.ravel().tolist(), aspect=20, orientation='vertical',pad=-.06)
    plt.colorbar(c, boundaries=np.linspace(0, 30, 11),ax=axes.ravel().tolist(), aspect=20, orientation='vertical',pad=0.04)
    if savefig:
        fig.savefig('../plots/phi_pdfs_ba_ca_flat.png', format='png', dpi=300);

        
def phi_pdfs_ba_ca_flat(agg_as_iceagg_flat, agg_bs_iceagg_flat, agg_cs_iceagg_flat, 
                        phi_ba_iceagg_flat, phi_ca_iceagg_flat,
                        phi_ba_aggagg_flat, phi_ca_aggagg_flat, savefig=False):


    fig, axes = plt.subplots(2,5, figsize=(16,5), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    axs = axes.ravel()

    phi_bins_flat=[0.005, 0.027, 0.039, 0.054, 0.072, 0.095,
     0.121, 0.151, 0.187, 0.229, 0.279, 0.337,
     0.398, 0.452, 0.498, 0.538, 0.576, 0.613,
     0.655, 0.710, 0.977]

    #r=9
    rs = [12, 12, 13, 13, 14, 15,15,15,15,15,15,15,14,13,12,11,11,10,10,10]
    i=0
    phios = [0,3,5,7,9,11,13,15,17,19]
    for c,phio in enumerate(phios):    
        r=rs[c]

        iceagg = np.vstack([phi_ba_iceagg_flat[phio,r,:], phi_ca_iceagg_flat[phio,r,:]])
        kde_iceagg = st.gaussian_kde(iceagg)    
        aggagg = np.vstack([phi_ba_aggagg_flat[phio,r,:], phi_ca_aggagg_flat[phio,r,:]])
        kde_aggagg = st.gaussian_kde(aggagg)

        # evaluate on a regular grid
        xgrid = np.arange(0.0, 1.01, 0.01)
        ygrid = np.arange(0.0, 1.01, 0.01)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z_iceagg = kde_iceagg.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_iceagg = Z_iceagg.reshape(Xgrid.shape)
        Z_aggagg = kde_aggagg.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Z_aggagg = Z_aggagg.reshape(Xgrid.shape)

        axs[i].contour(Z_iceagg, cmap='Blues', vmin=1, vmax=10)
        axs[i].contour(Z_aggagg, cmap='Reds', vmin=1, vmax=10)

        #modes iceagg
        modes = np.where(Z_iceagg==np.max(Z_iceagg))
        axs[i].plot(float(modes[1]), float(modes[0]), 'bo')
        #modes aggagg
        modes = np.where(Z_aggagg==np.max(Z_aggagg))
        axs[i].plot(float(modes[1]), float(modes[0]), 'ro')

        axs[i].plot(np.linspace(0.0,100,10),np.linspace(0.0,100,10), 'gray', '--', zorder=2)
        axs[i].set_xticks([0,20,40,60,80,100])
        axs[i].set_yticks([0,20,40,60,80,100])
        #axs[i].set_xticklabels([])
        axs[i].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], rotation=50)
        axs[i].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[i].grid(which='major', alpha=0.5)

        axs[i].set_title('[%.3f-%.3f]' %(phi_bins_flat[phio], phi_bins_flat[phio+1]),fontfamily='serif')

        i+=1
    m = plt.cm.ScalarMappable(cmap='Reds')
    m.set_array(Z_aggagg)

    c = plt.cm.ScalarMappable(cmap='Blues')
    c.set_array(Z_iceagg)

    cbar=plt.colorbar(m, boundaries=np.linspace(0, 50, 11),ax=axes.ravel().tolist(), aspect=20, orientation='vertical',pad=-.06)
    plt.colorbar(c, boundaries=np.linspace(0, 30, 11),ax=axes.ravel().tolist(), aspect=20, orientation='vertical',pad=0.04)
    if savefig:
        fig.savefig('../plots/phi_pdfs_ba_ca_flat.png', format='png', dpi=300);


