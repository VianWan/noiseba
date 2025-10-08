import numpy as np
import matplotlib.pyplot as plt 
from noiseba.preprocessing import stack_ccf

def plot_ccf(ccf_dict, ccf_distance, cor_time_begin,  dt, used_time=1, axes=None, plot_kwargs=None):
    CCF = []
    DIST = []
    for ccf_name, ccf_data in ccf_dict.items():
        cor_data = stack_ccf(ccf_data[0], method='linear', nu=2.0)
        dist = ccf_distance[ccf_name][0][0]
        CCF.append(cor_data)
        DIST.append(dist)

    ncf = np.r_[CCF]
    distance = np.array(DIST)
    ind = np.argsort(distance)
    ncf = ncf[ind]
    distance = distance[ind]

    ncf_start = cor_time_begin

    ind1 = int(((-used_time - ncf_start) / dt))
    ind2 = int((used_time - ncf_start) / dt)

    ncf = ncf[:, ind1:ind2]
    ncf /= max(np.max(ncf), 1e-8)
    t = np.arange(ncf.shape[1]) * dt - used_time
    print(ind1, ind2)
    
    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 7))

    if plot_kwargs is None:
        plot_kwargs = {}

    scale = 2
    for i in range(ncf.shape[0]):
        axes.plot(t, ncf[i] * scale + distance[i], color='k', **plot_kwargs)

    axes.set_xlabel('Lag time (s)', fontsize=24)
    axes.set_ylabel('Interstation distance (m)', fontsize=24)
    axes.set_xlim(-used_time, used_time)
    axes.grid(ls=':', lw=1.5, color='#AAAAAA')
    axes.tick_params(labelsize=12)
    plt.tight_layout()