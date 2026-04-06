import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

pWidth = 1.3333333 * 2.3
pHeight = 2.3

defcapsize=2

defpad = 10


def set_mpl():
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif')
    mpl.rc('font', size = '12')

    mpl.rc('xtick', labelsize=12) 
    mpl.rc('ytick', labelsize=12) 
    
    mpl.rc('lines', markersize = 2)
    
    
    mpl.rc('legend', fontsize = 10)
    
    mpl.rc('axes', titlesize = 12)
    mpl.rc('axes', titlepad = 10)


    mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

    
    
def errorplot(x,ymean,yerr,
                yfact=1,
                pyobj=plt,
                zorder=1,
                color="tab:green",
                label="",
                tMin=0,
                prune=1):
    pyobj.plot(x,
                     yfact * np.real(ymean), '-', zorder=zorder, color=color)
    pyobj.fill_between(
                x,
                (np.asarray(yfact * np.real(ymean)) -
                 np.asarray(yfact * np.real(yerr))),
                (np.asarray(yfact * np.real(ymean)) +
                 np.asarray(yfact * np.real(yerr))),
                linewidth=0,
                zorder=zorder-1,
                alpha=0.5, color=color, label=label)