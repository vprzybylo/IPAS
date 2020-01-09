"""Plots characteristic variables that have been looped over for every aspect ratio"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from ipas import lab_agg_agg_Ntot as lab
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from numpy.ma import masked_array
from matplotlib.ticker import MultipleLocator

class Make_Plots():

    def __init__(self, results):
        #probably a better way to unpack results after gathered 
        self.phio = np.array([results[x][0] for x in range(len(results))])
        self.r = np.array([results[x][1] for x in range(len(results))])
        self.width = np.array([results[x][2] for x in range(len(results))])
        self.length = np.array([results[x][3] for x in range(len(results))])
        self.rxs = np.array([results[x][4] for x in range(len(results))])
        self.rys = np.array([results[x][5] for x in range(len(results))])
        self.rzs = np.array([results[x][6] for x in range(len(results))])
        self.phi2Ds = np.array([results[x][7] for x in range(len(results))])
        self.cplxs =  np.array([results[x][8] for x in range(len(results))])
 


    def which_plot(self,nclusters,plot_name, ch_dist, savefile, read_file='outfile.dat', save=False, verbose=False, ext='eps'):
        #makes plots based on plot_name passed in

      
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator, LogFormatter, ScalarFormatter

        #self.numaspectratios = 26

        fig = plt.figure(frameon=False)
        ax = plt.subplot(111)

        #ax.set_xscale("log")
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
        ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')
        ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')

        plt.style.use('seaborn-dark')  #gray background
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        '''
        self.phip = self.phio[(self.phio<=1.0)]
        self.phic = self.phio[(self.phio>=1.0)]
        self.chp = self.chphi[(self.phio<=1.0)]
        self.chc = self.chphi[(self.phio>=1.0)]
        self.chp2D = self.chphi2D[(self.phio<=1.0)]
        self.chc2D = self.chphi2D[(self.phio>=1.0)]
        self.lengthp = self.length[(self.phio<=1.0)]
        self.widthp = self.width[(self.phio<=1.0)]
        self.lengthc = self.length[(self.phio>=1.0)]
        self.widthc = self.width[(self.phio>=1.0)]
        self.meanphip = self.mean_phi[(self.phio<1.0)]
        self.meanphic = self.mean_phi[(self.phio>=1.0)]
        self.meanphip2D = self.mean_phi2D[(self.phio<1.0)]
        self.meanphic2D = self.mean_phi2D[(self.phio>=1.0)]
        poserr_phip = self.poserr_phi[(self.phio<1.0)]
        negerr_phip = self.negerr_phi[(self.phio<1.0)]
        poserr_phic = self.poserr_phi[(self.phio>=1.0)]
        negerr_phic = self.negerr_phi[(self.phio>=1.0)]
        poserr_phip2D = self.poserr_phi2D[(self.phio<1.0)]
        negerr_phip2D = self.negerr_phi2D[(self.phio<1.0)]
        poserr_phic2D = self.poserr_phi2D[(self.phio>=1.0)]
        negerr_phic2D = self.negerr_phi2D[(self.phio>=1.0)]

        phiolog = np.log10(self.phio)
        phiplog = np.log10(self.phip)
        phiclog = np.log10(self.phic)
        '''

        if ch_dist=='gamma':
            dist_title = 'Gamma'
        else:
            dist_title = 'Best'

        if plot_name == 'char':

            '''
            #REGPLOT FIT
            p = sns.regplot(phip,chp,color='navy',scatter_kws={"s": 60},line_kws={'color':'darkorange'},lowess=False,label=label)
            q = sns.regplot(phic,chc,color='navy',scatter_kws={"s": 60},line_kws={'color':'darkorange'},lowess=True)
            slopep, interceptp, r_valuep, p_valuep, std_errp = stats.linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()
                                                                                [0].get_ydata())
            slopeq, interceptq, r_valueq, p_valueq, std_errq = stats.linregress(x=q.get_lines()[0].get_xdata(),y=q.get_lines()
                                                                                [0].get_ydata())
            '''
            #POLYFIT

            #coeffs = np.polyfit(phip,chp,4)
            #x2 = np.arange(min(phip)-1, max(phip), .01) #use more points for a smoother plot
            #y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2,color='darkorange',linewidth=2.5)  #best fit line
            plt.plot(self.phip, self.chp,'o',color='navy',markersize=6)
            plt.plot(self.phip, self.chp, color='navy')

            #coeffs = np.polyfit(phic,chc,4)
            #x2 = np.arange(min(phic), max(phic)+1, .01) #use more points for a smoother plot
            #y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2,color='darkorange',linewidth=2.5)
            plt.plot(self.phic, self.chc,'o',color='navy',markersize=6)
            plt.plot(self.phic, self.chc, color='navy', label = 'characteristic')
            #plt.plot(self.phio, self.mean_phi,'o',color='black',markersize=6)

            plt.plot(self.phip, self.meanphip, color='black')
            plt.plot(self.phic, self.meanphic, color='black', label = 'mean')
            ax.fill_between(self.phip, poserr_phip, negerr_phip, facecolor='darkorange', alpha=0.3)
            ax.fill_between(self.phic, poserr_phic, negerr_phic, facecolor='darkorange', alpha=0.3)


            #Line of no change in aspect ratio
            phiox=np.logspace(-2, 2., num=20, endpoint=True, base=10.0, dtype=None)#just columns (0,2); plates (-2,0)
            plt.ylim((.01,100))
            plt.xlim((.01,100))
            #plt.plot(phiox, phiox, color='navy')

            ax.set_yscale("log")
            ax.set_xscale("log")

            #plt.xlim((2.0,100))
            #plt.ylim((min(ch),max(ch)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
            #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

            #plt.title('Characteristic Aggregate Aspect Ratio for '+str(self.numaspectratios)+' Aspect Ratios')
            #plt.xlabel ("Equivalent Volume Aspect Ratio of Monomers")
            plt.title('Aggregate Aspect Ratio from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio")
            plt.legend(loc='best')

        if plot_name == '2D_phi':
            save=True

            self.phi2D=self.phi2D.reshape(20, 39)
            self.phioarr_ = np.array(self.phioarr)
            x = np.arange(len(self.phi2D[:,0]))
            y = np.arange(len(self.phi2D[0,:]))
            ylab=np.arange(2, len(self.phi2D[0,:]) +2)

            X_grid, Y_grid = np.meshgrid(x,y)
            
            #print(X_grid)
            
            Z_grid = self.phi2D[X_grid, Y_grid]
            
            xlab = self.phioarr_[X_grid]
            levels = np.logspace(np.amin(self.phi2D), np.amax(self.phi2D), 1)
            
            def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
                new_cmap = colors.LinearSegmentedColormap.from_list(
                    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                    cmap(np.linspace(minval, maxval, n)))
                return new_cmap


            cm = plt.cm.get_cmap('hot', 10)
            new_cmap = truncate_colormap(cm, 0.0, 0.87)
            im = plt.scatter(xlab, ylab[Y_grid], c=Z_grid, cmap=new_cmap, vmin = 0.0, vmax = 0.65)
            
            cb = plt.colorbar(im, orientation='vertical', label='2D Aspect Ratio',
                               shrink=0.8, format='%.3f')
            cb.outline.set_edgecolor('black')
            cb.outline.set_linewidth(1)
            tick_locator = ticker.MaxNLocator(nbins=9)
            cb.locator = tick_locator
            cb.update_ticks()
            #print(Z_grid)
            #cb = fig.colorbar(im, ax=ax)
            #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            #xlabels=np.logspace(-2.0,2.0, 20)
            
            ax.set_xlim(0.009,110.)
            xlabels = [.01,.10,1.0,10.,100.]
            ax.set_xticklabels(xlabels)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(FormatStrFormatter("%5f"))
            ax.set_xscale("log")

            #plt.title('Characteristic Eq. Volume Radius from the %s Distribution' %dist_title)
            ax.yaxis.label.set_size(14)
            ax.xaxis.label.set_size(14)
            #plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.xaxis.labelpad = 10
            #ax.yaxis.labelpad = 10
            
            ax.set_xlabel('Monomer Aspect Ratio')
            ax.set_ylabel('# of Monomers in Aggregate')

            plt.show()

        if plot_name == 'shape':

            #REGPLOT FIT

            #p = sns.regplot(phio,shape,color='navy',scatter_kws={"s": 80},\
            #    line_kws={'color':'darkorange'},lowess=True, label=label)
            #slopep, interceptp, r_valuep, p_valuep, std_errp = stats.linregress(x=p.get_lines()[0].get_xdata(),\
            #                y=p.get_lines()[0].get_ydata())

            #POLYFIT
            '''

            coeffs = np.polyfit(phio,shape,2)
            x2 = np.arange(min(phio)-1, max(phio), .01) #use more points for a smoother plot
            y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2,color='darkorange',linewidth=2.5)
            '''
            if ch_dist == 'gamma':

                titlearr = ['$\phi$ ', 'Equiv. Vol. ', 'Depth ', 'Major Axis ']
                filearr = ['phi', 'Req', 'Depth', 'MjrAx']

                shapearr = [[] for i in range(len(titlearr))]
                shapearr[0]=phishape
                shapearr[1]=reqshape
                shapearr[2]=depthshape
                shapearr[3]=majoraxshape

                for i in range(0,len(titlearr)):

                    fig = plt.figure(figsize=(10,10))
                    ax = plt.subplot(111)
                    ax.set_xscale("log")
                    #ax.set_yscale("log")
                    ax.grid(b=True, which='minor', alpha = 0.9, axis = 'x')
                    ax.grid(b=True, which='minor', alpha = 0.9, axis = 'y')
                    ax.grid(b=True, which='major', alpha = 1.0, axis = 'x')
                    ax.grid(b=True, which='major', alpha = 1.0, axis = 'y')

                    plt.plot(self.phio,shapearr[i],'o',color='navy',markersize=6)
                    plt.title('%s Shape Parameter from %s Distribution' %(titlearr[i], dist_title))
                    #plt.title('Shape Parameter for '+str(self.numaspectratios)+' Aspect Ratios')
                    plt.ylabel("Shape of %s Distribution" %titlearr[i])
                    plt.xlabel ("Equivalent Volume Aspect Ratio of Monomer")
                    ax.set_yscale("log")
                    ax.set_xscale("log")
                    plt.ylim(min(shapearr[i]),max(shapearr[i])+20)
                    #plt.ylim(min(shape),10000)
                    plt.xlim((.009,110))
                    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
                    #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))


                    if save:
                        savefile = '%s_shape' %filearr[i]

                        filename = savefile

                        self.save_fig(nclusters, filename=filename, ext=ext, verbose = verbose)



            else:
                print('Shape parameter plot is only applicable to the gamma distribution')


        if plot_name == 'dphigamquad':

            #Quadrant Plot

            dphi_neg_plate = self.dphigam[(self.phio<1.0) & (self.dphigam<0.0)]
            phio_neg_plate = self.phio[(self.phio<1.0) & (self.dphigam<0.0)]
            dphi_pos_plate = self.dphigam[(self.phio<1.0) & (self.dphigam>0.0)]
            phio_pos_plate = self.phio[(self.phio<1.0) & (self.dphigam>0.0)]
            dphi_pos_col = self.dphigam[(self.phio>1.0) & (self.dphigam>0.0)]
            phio_pos_col = self.phio[(self.phio>1.0) & (self.dphigam>0.0)]
            dphi_neg_col = self.dphigam[(self.phio>1.0) & (self.dphigam<0.0)]
            phio_neg_col = self.phio[(self.phio>1.0) & (self.dphigam<0.0)]


            ax1 = plt.subplot(211)
            ax1.grid(b=True, which='minor', alpha = 0.5, axis = 'y')
            ax1.grid(b=True, which='minor', alpha = 0.5, axis = 'x')
            ax1.grid(b=True, which='major', alpha = 0.9, axis = 'x')
            ax1.grid(b=True, which='major', alpha = 0.9, axis = 'y')
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            plt.title('Characteristic Change in $\phi$ from %s Distribution' %dist_title)
            fig.text(0.03, 0.5, 'Aggregate $\phi$ - Monomer $\phi$',
                     va='center', rotation='vertical', fontsize=12)
            #maxc = max(dphi_pos_col)
            #maxp = max(dphi_pos_plate)
            #plt.ylim(.011, max(maxc, maxp)+.1)
            plt.setp(ax1.get_xticklabels(), fontsize=12)
            #ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            '''
            plt.plot(phio_pos_plate,dphi_pos_plate, color='navy', marker='o', markersize=7)
            plt.plot(phio_pos_col,dphi_pos_col, color='navy', marker='o', markersize=7)
            plt.axhline(y=0.0112, color='black', linestyle='-',linewidth=2.5)
            plt.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)
            #plt.legend(loc='best')

            # share x only
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.grid(b=True, which='minor', alpha = 0.5, axis = 'y')
            ax2.grid(b=True, which='minor', alpha = 0.5, axis = 'x')
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'x')
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'y')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel ("Monomer Aspect Ratio")
            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

            plt.gca().invert_yaxis()
            plt.plot(phio_neg_plate,abs(dphi_neg_plate), color='navy', marker='o', markersize=7)
            plt.plot(phio_neg_col,abs(dphi_neg_col), color='navy', marker='o', markersize=7)
            plt.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)

            #plt.ylim(max(abs(dphi_neg_col)+100),min(abs(dphi_neg_plate)))
            plt.subplots_adjust(hspace=0)
            '''

            ax1.plot(phio_pos_plate,dphi_pos_plate, color='navy', marker='o', markersize=7)
            ax1.plot(phio_pos_col,dphi_pos_col, color='navy', marker='o', markersize=7)
            plt.axhline(y=0.001, color='black', linestyle='-',linewidth=2.5)
            ax1.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)

            #plt.legend(loc='best')

            # share x only
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.grid(b=True, which='minor', alpha = 0.5, axis = 'y')
            ax2.grid(b=True, which='minor', alpha = 0.5, axis = 'x')
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'x')
            ax2.grid(b=True, which='major', alpha = 0.9, axis = 'y')
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel ("Monomer Aspect Ratio")
            ax2.axvline(x=1.0, color='black', linestyle='-',linewidth=2.5)

            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            #ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

            #ax1.invert_yaxis()
            #ax2.invert_yaxis()
            ax2.plot(phio_neg_plate,abs(dphi_neg_plate), color='navy', marker='o', markersize=7)
            ax2.plot(phio_neg_col,abs(dphi_neg_col), color='navy', marker='o', markersize=7)
            ax1.set_ylim(.001,1000)
            ax2.set_ylim(1000,.001)
            plt.subplots_adjust(hspace=0)

        if plot_name == 'dphigamW':
            #W Plot
            plt.plot(self.phio,abs(self.dphigam),color='navy')
            plt.plot(self.phio,abs(self.dphigam), color='navy',marker='o',markersize=5)
            ax.set_yscale("log")
            #plt.title('Change in $\phi$ for '+str(self.numaspectratios)+' Aspect Ratios')
            plt.title('Change in eq. vol. $\phi$ from Characteristic of %s Distribution' %dist_title)
            ax.set_xscale("log")
            ax.set_xlabel ("Equivalent Volume Monomer Aspect Ratio")
            ax.set_ylabel('Aggregate $\phi$ - Monomer $\phi$')
            #plt.legend(loc='best')
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

        if plot_name == 'overlap':

            '''
            self.stddevpos_ovrlp = self.poserr_ovrlp - self.mean_ovrlp
            self.stddevneg_ovrlp = self.negerr_ovrlp - self.mean_ovrlp
            cv_pos = self.stddevpos_ovrlp/self.mean_ovrlp
            cv_neg = self.stddevneg_ovrlp/self.mean_ovrlp

            self.ch_ovrlpp = self.ch_ovrlp[(self.phio<1.0)]
            poly = pl.polyfit(phiplog, self.ch_ovrlpp, 2)  #2 is the degree of polynomial to fit
            x2 = np.arange(min(phiplog), max(phiplog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line

            self.ch_ovrlpc = self.ch_ovrlp[(self.phio>=1.0)]
            poly = pl.polyfit(phiclog, self.ch_ovrlpc, 2)  #2 is the degree of polynomial to fit
            x2 = np.arange(min(phiclog), max(phiclog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            '''
            xstar = np.arange(0,50)
            plt.plot(xstar,self.ch_ovrlp[:,1],'*',color='blue',markersize=6, label='characteristic')
            #plt.plot(phiolog,self.ch_ovrlp,color='navy')

            #ax.set_xscale("log")
            #plt.xlim((.01,100))
            #plt.ylim((min(overlap),10))

            ovrlps = [np.array(i) for i in self.ovrlp[:,:,1]]
            phio_formatted = [ '%.2f' % elem for elem in self.phio ]

            #for i in range(len(phio_formatted)):
            #    print(i)
            #    if i != 0 and i != 9 and i != 19 and i != 29 and i != 39 and i != 49:
            #        print('here')
            #        phio_formatted[i]=''

            plt.xticks(rotation=45)
            plt.boxplot(ovrlps)
            ax.xaxis.set_ticklabels(phio_formatted)

            #ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

            #ax.fill_between(self.phio, self.poserr_ovrlp, self.negerr_ovrlp, facecolor='darkorange', alpha=0.5)
            #ax.fill_between(self.phio,  self.mean_ovrlp+cv_pos,\
            #                self.mean_ovrlp+cv_neg, facecolor='darkorange', alpha=0.5)
            plt.legend(loc='best')

            plt.title('Overlap from the %s Distribution' %dist_title)
            plt.ylabel ("Overlap [%]")
            #plt.xlim((2.0,100))

        if plot_name == 'phi_contour':
            save=True

            self.ch_phi=self.chphi.reshape(20,39)
            self.ch_phi=self.ch_phi[:,1:]
            
            r=1
            #print('mean plates', np.average(self.ch_phi[r,0:10,8]))
            #print('mean columns', np.average(self.ch_phi[r,10:,8]))
            self.phioarr_ = np.array(self.phioarr).reshape(28,20)[0]
            #self.phioarr_ = np.array(self.phioarr)
            #print(self.phioarr_)
            
            ch_plates = self.ch_phi[:int(len(self.phioarr_)/2),:]   
            
            ch_col = self.ch_phi[int((len(self.phioarr_)/2)):,:]
            print(int((len(self.phioarr_)/2)))
            print(np.shape(self.ch_phi))
            #print('col max min', np.max(ch_col), np.min(ch_col))
            #print('plate min max', np.min(ch_plates), np.max(ch_plates))
            #print(np.shape(ch_plates))
            #print(np.shape(ch_col))
            
            xp = np.arange(int(len(self.ch_phi[:,0])/2))
            #print('xp', len(xp))
            xc = np.arange(10,int(len(self.ch_phi[:,0])))
            #print('xc', len(xc))

            #print(len(self.phioarr_))
            y = np.arange(len(self.ch_phi[0,:]))
            
            ylab=np.arange(1, len(self.ch_phi[0,:]) +1)

            X_gridp, Y_grid = np.meshgrid(xp,y)
            X_gridc, Y_grid = np.meshgrid(xc,y)
            
            #print(len(y))
            
            Z_grid_p = ch_plates[X_gridp, Y_grid]
            Z_grid_c = ch_col[X_gridp, Y_grid]
            
            #xlabp = self.phioarr_[:int(X_grid/2)-1]            
            #xlabc = self.phioarr_[int(X_grid/2):]
            #print(xlabc)
        
            levels = np.logspace(np.amin(self.ch_phi), np.amax(self.ch_phi), 1)
            maxch = np.amax(self.ch_phi)
            minch = np.amin(self.ch_phi)

            #cbar = ax.fig.colorbar(, ax=ax, **cbar_kw)
                   
            
            def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
                new_cmap = colors.LinearSegmentedColormap.from_list(
                    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                    cmap(np.linspace(minval, maxval, n)))
                return new_cmap

            bins=20
            cm_b = plt.cm.get_cmap('ocean', bins)
            cm_a = plt.cm.get_cmap('hot', bins)     
            new_cmap_b = truncate_colormap(cm_b, 0.87, 0.0)
            new_cmap_a = truncate_colormap(cm_a, 0.0, 0.87)

            #cm_b = plt.cm.get_cmap('viridis_r', 8)      
                       
            im_a = plt.scatter(self.phioarr_[X_gridp], ylab[Y_grid], c=Z_grid_p, cmap=new_cmap_a)
            im_b = plt.scatter(self.phioarr_[X_gridc], ylab[Y_grid], c=Z_grid_c, cmap=new_cmap_b)
            #cbar = fig.colorbar(mappable=im_b, ax=ax)            
    
            plt.title('Characteristic Aggregate Aspect Ratio')
            
            cbar_b = plt.colorbar(im_b, orientation='vertical', cmap=new_cmap_b, shrink=0.8, format='%.1f',drawedges=True)
            #cbar_b.locator = MultipleLocator(1) # Show ticks only for each multiple of 1
            ticks_b = np.linspace(np.max(ch_col), np.min(ch_col), bins-10)
            cbar_b.set_ticks(ticks_b, update_ticks=True)
            
            cbar_b.set_clim(vmin=np.min(ch_col), vmax=np.max(ch_col))
            cbar_b.outline.set_edgecolor('black')
            cbar_b.outline.set_linewidth(1)
            cbar_b.ax.invert_yaxis() 
            
            cbar_a = plt.colorbar(im_a, orientation='vertical', cmap=new_cmap_a, shrink=0.8, format='%.2f', drawedges=True)
            ticks_a = np.linspace(np.min(ch_plates), np.max(ch_plates),bins-10)
            cbar_a.set_ticks(ticks_a, update_ticks=True)
            cbar_a.set_clim(vmin=np.min(ch_plates), vmax=np.max(ch_plates))
            cbar_a.outline.set_edgecolor('black')
            cbar_a.outline.set_linewidth(1)
            #cbar_a.ax.invert_yaxis() 
            
            ax.set_xlim(0.009,110.)
            xlabels = [.01,.10,1.0,10.,100.]
            ax.set_xticklabels(xlabels)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(FormatStrFormatter("%5f"))
            ax.set_xscale("log")

            #plt.title('Characteristic Eq. Volume Radius from the %s Distribution' %dist_title)
            ax.yaxis.label.set_size(14)
            ax.xaxis.label.set_size(14)
            #plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.xaxis.labelpad = 10
            #ax.yaxis.labelpad = 10
            
            ax.set_xlabel('Monomer Aspect Ratio')
            ax.set_ylabel('# of Monomers in Aggregate')
            
            #plt.show()

       
        if plot_name == 'overlap_contour':
            nclusters = 300
            x = np.arange(len(self.phio))
            y = np.arange(len(self.ch_ovrlp[0,:]))
            ylab=np.arange(1, len(self.ch_ovrlp[0,:]) +1)

            X_grid, Y_grid = np.meshgrid(x,y)
            Z_grid = self.ch_ovrlp[X_grid,Y_grid]
            xlab = self.phio[X_grid]
            Z_grid = self.ch_ovrlp[X_grid,Y_grid]
            maxch = np.amax(self.ch_ovrlp)
            minch = np.amin(self.ch_ovrlp)
            labels = np.arange(minch, maxch, 5)
            levels = np.arange(minch, maxch, 2)
            print('overlap', minch, maxch)
            xlabels = [.01,.10,1.0,10,100]
            ax.set_xticklabels(xlabels)

            #ax.set_xticks(levels)
            colormap = 'nipy_spectral'
            im = ax.contourf(np.log10(xlab), ylab[Y_grid], Z_grid,
                             cmap=colormap, levels=levels)

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Eq. Volume Monomer Aspect Ratio')
            ax.set_ylabel('# of Monomers in Aggregate')
            ax.yaxis.label.set_size(14)
            ax.xaxis.label.set_size(14)
            plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.set_zlabel('Separation Ratio, S~0 => centers close')
            #ax.xaxis.labelpad = 10
            #ax.yaxis.labelpad = 10

            CBI = plt.colorbar(im, orientation='vertical', label='Overlap [%]',
                               shrink=0.8, format='%1.1f', ticks=labels)


        if plot_name == 'S_contour':
            x = np.arange(len(self.phio))
            y = np.arange(len(self.ch_S[0,:]))
            ylab=np.arange(1, len(self.ch_S[0,:]) +1)

            X_grid, Y_grid = np.meshgrid(x,y)
            xlab = self.phio[X_grid]

            Z_grid = self.ch_S[X_grid,Y_grid]
            maxch = np.amax(self.ch_S)
            minch = np.amin(self.ch_S)
            print('S', minch, maxch)
            labels = [.01,.05,.10,.15,.20,.25,.30,.35]
            levels = np.arange(minch, maxch, .01)

            im = plt.contourf(np.log10(xlab), ylab[Y_grid], Z_grid, cmap = 'nipy_spectral',
                              vmin = .01, vmax=.40,levels=levels)

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            #plt.title('Characteristic S parameter from the %s Distribution' %dist_title)
            ax.yaxis.label.set_size(14)
            ax.xaxis.label.set_size(14)
            plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.xaxis.labelpad = 10
            #ax.yaxis.labelpad = 10
            ax.set_xlabel('Eq. Volume Monomer Phi')
            ax.set_ylabel('# of Monomers in Aggregate')

            xlabels = [.01,.10,1.0,10,100]
            ax.set_xticklabels(xlabels)
            CBI = plt.colorbar(im, orientation='vertical', label='Separation Ratio',
                               shrink=0.8, format='%1.2f', ticks=labels)


        if plot_name == 'dd_contour':
            
            x = np.arange(len(self.phioarr))
            y = np.arange(len(self.ch_dd[0,:]))
            ylab=np.arange(1, len(self.ch_dd[0,:])+1)

            X_grid, Y_grid = np.meshgrid(x,y)
            xlab = self.phio[X_grid]

            Z_grid = self.ch_dd[X_grid,Y_grid]
            maxch = np.amax(self.ch_dd)
            minch = np.amin(self.ch_dd)
            print('dd', minch, maxch)
            labels = np.arange(minch, maxch, 0.25)
            labels = [-.05, 0.0, 0.05, .1, .2, .3, .4,.5,.6,.7,.8,.9, 1.0]
            levels = np.arange(minch, maxch, 0.05)
            #print(levels)

            colormap = 'nipy_spectral'
            im = ax.contourf(np.log10(xlab), ylab[Y_grid], Z_grid,
                             cmap=colormap, levels=levels)

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Eq. Volume Monomer Aspect Ratio')
            ax.set_ylabel('# of Monomers in Aggregate')
            ax.yaxis.label.set_size(14)
            ax.xaxis.label.set_size(14)
            plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.xaxis.labelpad = 10
            #ax.yaxis.labelpad = 10

            xlabels = [.01,.10,1.0,10,100]
            ax.set_xticklabels(xlabels)
            CBI = plt.colorbar(im, orientation='vertical', label='Change in Density',
                               shrink=0.8, format='%1.2f', ticks=labels)
            
        if plot_name == 'dd_discrete':
            self.ch_dd=self.ch_dd.reshape((10, 19))
            
            #plt.style.use(['seaborn-white','seaborn-paper'])
            x = np.arange(len(self.phioarr))
            maxch = np.amax(self.ch_dd)
            minch = np.amin(self.ch_dd)
            ax.set_xscale("log")
            ax.set_xlim(0.01,100.000)
            for i in range(6):
                ax.plot(self.phioarr, self.ch_dd[:,i]*100,label=(str(i+2) +' monomers'))
            plt.legend()

            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            ax.set_xlabel('Monomer Aspect Ratio')
            ax.set_ylabel('% Decrease in Density')

            plt.show()


        if plot_name == 'req_contour':
            save=True

            self.chreq=self.ch_req.reshape(20, 39)
            self.chreq=self.chreq[:,:]
            self.phioarr_ = np.array(self.phioarr)
            x = np.arange(len(self.chreq[:,0]))
            y = np.arange(len(self.chreq[0,:]))
            ylab=np.arange(2, len(self.chreq[0,:]) +2)

            X_grid, Y_grid = np.meshgrid(x,y)
            
            #print(X_grid)
            
            Z_grid = self.chreq[X_grid, Y_grid]
            
            xlab = self.phioarr_[X_grid]
            levels = np.logspace(np.amin(self.chreq), np.amax(self.chreq), 1)
            
            cm = plt.cm.get_cmap('CMRmap', 10)
            im = plt.scatter(xlab, ylab[Y_grid], c=Z_grid, cmap=cm)
            
            cb = plt.colorbar(im, orientation='vertical', label='Characteristic Aggregate Radius',
                               shrink=0.8, format='%.0f')
            cb.outline.set_edgecolor('black')
            cb.outline.set_linewidth(1)
            tick_locator = ticker.MaxNLocator(nbins=7)
            cb.locator = tick_locator
            cb.update_ticks()
            #print(Z_grid)
            #cb = fig.colorbar(im, ax=ax)
            #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            #xlabels=np.logspace(-2.0,2.0, 20)
            
            ax.set_xlim(0.009,110.)
            xlabels = [.01,.10,1.0,10.,100.]
            ax.set_xticklabels(xlabels)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(FormatStrFormatter("%5f"))
            ax.set_xscale("log")

            #plt.title('Characteristic Eq. Volume Radius from the %s Distribution' %dist_title)
            ax.yaxis.label.set_size(14)
            ax.xaxis.label.set_size(14)
            #plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.xaxis.labelpad = 10
            #ax.yaxis.labelpad = 10
            
            ax.set_xlabel('Monomer Aspect Ratio')
            ax.set_ylabel('# of Monomers in Aggregate')

            plt.show()

        if plot_name == 'S':

            xstar = np.arange(0,50)
            plt.plot(xstar,self.ch_S[:,1],'*',color='blue',markersize=6, label='characteristic')
            #plt.plot(phiolog,self.ch_ovrlp,color='navy')

            #ax.set_xscale("log")

            Ss = [np.array(i) for i in self.S[:,:,1]]

            phio_formatted = [ '%.2f' % elem for elem in self.phio ]

            #for i in range(len(phio_formatted)):
            #    print(i)
            #    if i != 0 and i != 9 and i != 19 and i != 29 and i != 39 and i != 49:
            #        print('here')
            #        phio_formatted[i]=''

            plt.xticks(rotation=45)
            plt.boxplot(Ss)
            ax.xaxis.set_ticklabels(phio_formatted)

            #ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

            #ax.fill_between(self.phio, self.poserr_ovrlp, self.negerr_ovrlp, facecolor='darkorange', alpha=0.5)
            #ax.fill_between(self.phio,  self.mean_ovrlp+cv_pos,\
            #                self.mean_ovrlp+cv_neg, facecolor='darkorange', alpha=0.5)
            plt.legend(loc='best')

            #plt.title('Separation Ratio from the %s Distribution' %dist_title)
            plt.ylabel ("S")
            #plt.xlim((2.0,100))

        if plot_name == 'complexity':
            save=True

            self.chcplx=self.chcplx.reshape(28, 20, 38)
            
            self.chcplx=self.chcplx[:,:,:]
            maxch = np.amax(self.chcplx)
            minch = np.amin(self.chcplx)
            print(maxch, minch)
            self.phioarr_ = np.array(self.phioarr).reshape(28,20)[0]
            x = np.arange(len(self.chcplx[0,:,0]))
            y = np.arange(len(self.chcplx[0,0,:]))
            ylab=np.arange(2, len(self.chcplx[0,0,:]) +2)

            X_grid, Y_grid = np.meshgrid(x,y)
            
            #print(X_grid)
            
            Z_grid = self.chcplx[20, X_grid, Y_grid]
            
            xlab = self.phioarr_[X_grid]
            levels = np.logspace(np.amin(self.chcplx), np.amax(self.chcplx), 1)
            
            cm = plt.cm.get_cmap('jet', 10)
            im = plt.scatter(xlab, ylab[Y_grid], c=Z_grid, cmap=cm, vmin=0.0, vmax = 1.0)
            
            cb = plt.colorbar(im, orientation='vertical', label='Characteristic Aggregate Radius',
                               shrink=0.8, format='%.3f')
            #tick_locator = ticker.MaxNLocator(nbins=7)
            #cb.locator = tick_locator
            #cb.update_ticks()
            #print(Z_grid)
            #cb = fig.colorbar(im, ax=ax)
            #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            #xlabels=np.logspace(-2.0,2.0, 20)
            
            ax.set_xlim(0.009,110.)
            xlabels = [.01,.10,1.0,10.,100.]
            ax.set_xticklabels(xlabels)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(FormatStrFormatter("%5f"))
            ax.set_xscale("log")

            #plt.title('Characteristic Eq. Volume Radius from the %s Distribution' %dist_title)
            ax.yaxis.label.set_size(14)
            ax.xaxis.label.set_size(14)
            #plt.tick_params(axis='both', which='major', labelsize=14)
            #ax.xaxis.labelpad = 10
            #ax.yaxis.labelpad = 10
            
            ax.set_xlabel('Monomer Aspect Ratio')
            ax.set_ylabel('# of Monomers in Aggregate')

            #plt.show()


        if plot_name == 'req':

            agg_mass = (.9*4/3*np.pi*10**3)*2
            agg_rho = []
            for i in self.chreq:
                agg_rho.append(agg_mass/(4/3*np.pi*i**3))

            plt.xlim((.01,100))
            #print(agg_rho)
            ax.set_xscale("log")
            #ax.set_ylim((0,max(self.chreq)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            ax.fill_between(self.phio, self.poserr_req, self.negerr_req, facecolor='darkorange', alpha=0.5)
            ax.plot(self.phio,self.chreq,'o',color='navy',markersize=6)
            ax.plot(self.phio,self.chreq,color='navy')
            ax.axhline(y=10, color='navy', linestyle='--')

            plt.title('Characteristic Eq. Volume Radius from %s Distribution' %dist_title)
            ax.set_ylabel ("Aggregate Eq. Volume Radius")

        if plot_name == 'major_axis':

            ax.set_yscale("log")
            ax.set_xscale("log")

            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))
            ax.fill_between(self.phio, self.negerr_mjrax, self.poserr_mjrax, facecolor='#0000e6', alpha=0.3)
            ax.fill_between(self.phio, self.negerr_depth, self.poserr_depth, facecolor='#EE7600', alpha=0.3)
            plt.plot(self.phio,self.mean_mjrax, color='black')
            plt.plot(self.phio,self.ch_majorax,'o',color='lightblue',markersize=3,label = 'Major axis from ellipse')
            plt.plot(self.phio,self.ch_majorax, color='lightblue')
            plt.plot(self.phio,self.mean_depth, color='black')
            plt.plot(self.phio,self.ch_depth,'o',color='#ffc266',markersize=3, label = 'Depth')
            plt.plot(self.phio,self.ch_depth,color='#ffc266')
            plt.plot(self.phip,self.chp,'o',color='#adebad',markersize=3, label = 'Aggregate aspect ratio')
            plt.plot(self.phip,self.chp, color='#adebad')
            plt.plot(self.phic,self.chc,'o',color='#adebad',markersize=3)
            plt.plot(self.phic,self.chc, color='#adebad')
            plt.plot(self.phip,self.meanphip, color='black')
            plt.plot(self.phic,self.meanphic, color='black')
            ax.fill_between(self.phip, poserr_phip, negerr_phip, facecolor='darkgreen', alpha=0.3)
            ax.fill_between(self.phic, poserr_phic, negerr_phic, facecolor='darkgreen', alpha=0.3)
            plt.plot(self.phip,self.lengthp,color='darkorange', linestyle = '--')
            plt.plot(self.phic,self.lengthc,color='navy', linestyle = '--')
            plt.plot(self.phip,self.widthp,color='navy',linestyle = '--')
            plt.plot(self.phic,self.widthc,color='darkorange',linestyle = '--')

            plt.title('Characteristic Values from %s Distribution' %dist_title)
            plt.ylabel ("Aggregate Aspect Ratio, Major Axis, and Depth")
            plt.legend(loc='best')

        if plot_name == 'dc_da':
            #ax.set_yscale("log")
            ax.set_xscale("log")

            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))

            ch_majoraxp = self.ch_majorax[(self.phio<1.0)]
            ch_majoraxc = self.ch_majorax[(self.phio>=1.0)]
            ch_depthp = self.ch_depth[(self.phio<1.0)]
            ch_depthc = self.ch_depth[(self.phio>=1.0)]
            da_p = ch_majoraxp - self.widthp
            dc_p = ch_depthp - self.lengthp
            da_c = ch_depthc - self.widthc
            dc_c = ch_majoraxc - self.lengthc
            dc_da_p = dc_p/da_p
            dc_da_c = dc_c/da_c

            plt.plot(self.phip,da_p,'o',color='navy',markersize=3,label = 'da')
            plt.plot(self.phip,da_p,color='navy')
            #plt.plot(phip,dc_da_p,'o',color='darkgreen',markersize=3)
            #plt.plot(phip,dc_da_p,color='darkgreen')
            #plt.plot(phic,dc_da_c,'o',color='darkgreen',markersize=3,label = 'dc/da')
            #plt.plot(phic,dc_da_c,color='darkgreen')
            plt.plot(self.phip,dc_p,'o',color='darkorange',markersize=3,label = 'dc')
            plt.plot(self.phip,dc_p,color='darkorange')
            plt.plot(self.phic,da_c,'o',color='navy',markersize=3)
            plt.plot(self.phic,da_c,color='navy')
            plt.plot(self.phic,dc_c,'o',color='darkorange',markersize=3)
            plt.plot(self.phic,dc_c,color='darkorange')
            #plt.axhline(y=1, color='black', linestyle='-',linewidth=2.5)

            plt.ylabel ("Aggregate c (a) - Monomer c (a)")
            plt.legend(loc='best')

        if plot_name == 'dc_da_normalized':

            ax.set_yscale("log")
            plt.xlim((.01,100))
            #plt.ylim((min(major_axis),10))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3.2f"))

            ch_majoraxp = self.ch_majorax[(self.phio<1.0)]
            ch_majoraxc = self.ch_majorax[(self.phio>=1.0)]
            ch_depthp = self.ch_depth[(self.phio<1.0)]
            ch_depthc = self.ch_depth[(self.phio>=1.0)]
            da_p = (ch_majoraxp - self.widthp)/self.widthp
            dc_p = (ch_depthp - self.lengthp)/self.lengthp
            da_c = (ch_depthc - self.widthc)/self.widthc
            dc_c = (ch_majoraxc - self.lengthc)/self.lengthc

            plt.plot(self.phip,da_p,'o',color='navy',markersize=3,label = 'da')
            plt.plot(self.phip,da_p,color='navy')
            #plt.plot(phip,dc_da_p,'o',color='darkgreen',markersize=3)
            #plt.plot(phip,dc_da_p,color='darkgreen')
            #plt.plot(phic,dc_da_c,'o',color='darkgreen',markersize=3,label = 'dc/da')
            #plt.plot(phic,dc_da_c,color='darkgreen')
            plt.plot(self.phip,dc_p,'o',color='darkorange',markersize=3,label = 'dc')
            plt.plot(self.phip,dc_p,color='darkorange')
            plt.plot(self.phic,da_c,'o',color='navy',markersize=3)
            plt.plot(self.phic,da_c,color='navy')
            plt.plot(self.phic,dc_c,'o',color='darkorange',markersize=3)
            plt.plot(self.phic,dc_c,color='darkorange')
            #plt.axhline(y=1, color='black', linestyle='-',linewidth=2.5)

            plt.title('Normalized Change in Characteristic Axes Lengths from the %s Distribution' %dist_title)
            plt.ylabel ("(Aggregate c (a) - Monomer c (a))/ Monomer c (a)")
            plt.legend(loc='best')
            ax.set_xscale("log")

        if plot_name == 'max_contact_angle':

            import random
            xrotp = self.xrot[(self.phio<=1.0)]*180/np.pi
            yrotp = self.yrot[(self.phio<=1.0)]*180/np.pi
            yrotc = self.yrot[(self.phio>=1.0)]
            rotplate = [max(i, j) for i, j in zip(xrotp, yrotp)]
            rotcol = yrotc*180/np.pi * 2
            rot = np.concatenate((rotplate,rotcol))

            poly = pl.polyfit(phiplog, rotplate, 3)  #2 is the degree of polynomial to fit
            x2 = np.arange(min(phiplog), max(phiplog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            #print('poly plates = ', poly)

            poly = pl.polyfit(phiclog, rotcol, 3)
            x2 = np.arange(min(phiclog), max(phiclog), .01) #use more points for a smoother plot
            y2 = np.polyval(poly, x2) #Evaluates the polynomial for each x2 value
            #plt.plot(x2, y2, color='darkorange', linewidth=5)  #best fit line
            #print('poly columns = ', poly)

            #plt.xlim((.01,100))
            #plt.ylim((0,max(rotplate, rotcol)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%3d"))

            plt.plot(self.phio,rot,'o',color='navy',markersize=6)
            plt.plot(self.phio,rot,color='navy')
            #plt.plot(self.phic,rotcol,'o',color='navy',markersize=6)
            #plt.plot(self.phic,rotcol,color='navy')
            plt.title('Maximum Possible Contact Angle')
            plt.ylabel ("Aggregate Contact Angle [degrees]")
            #ax.set_xscale("log")


        ax.set_xlabel ("Equivalent Volume Aspect Ratio of Monomer")


        if plot_name != 'shape' and save:

            filename = savefile

            self.save_fig(nclusters, filename=filename, ext=ext, verbose = verbose)
            #plt.show()
        else:
            plt.close()



    def save_fig(self, nclusters, filename, ext='eps', close=True, verbose=False):

        """Save a figure from pyplot.

        Parameters
        ----------
        path : string
            The path without the extension to save the
            figure to.

        filename : string
            name of the file to save

        ext : string (default='png')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

        close : boolean (default=True)
            Whether to close the figure after saving.  If you want to save
            the figure multiple times (e.g., to multiple formats), you
            should NOT close it in between saves or you will have to
            re-plot it.

        verbose : boolean (default=True)
            Whether to print information about when and where the image
            has been saved.

            """
        import os.path


        path=('/network/rit/lab/sulialab/share/IPAS_lookup/39crystals/300xtals_hist/depth/')

        filename = "%s.%s" % (filename, ext)
        if path == '':
            path = '.'

        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

        # The final path to save to
        savepath = os.path.join(path, filename)

        if verbose:
            print("Saving figure to '%s'..." % savepath),

        # Actually save the figure
        plt.savefig(savepath)

        # Close it
        if close:
            plt.close()
