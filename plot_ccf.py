from obspy import read
import matplotlib.pyplot as plt
import numpy as np
import sys

argc = len(sys.argv)
if argc != 2:
    print('Usage: python %s ccf_dir'%sys.argv[0])
    exit(1)

ccf_dir = sys.argv[1]
if ccf_dir[-1] != '/':
    ccf_dir += '/'

st = read(ccf_dir + 'COR*sac')

f1 = 0.1; f2 = 15
#st.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=True)
t0 = 4
t1 = -t0; t2 = t0
scale = 2
ccf = []; dist = []
for i, tr in enumerate(st):
    b = tr.stats.sac.b
    # x1, y1 = tr.stats.sac.evlo, tr.stats.sac.evla
    # x2, y2 = tr.stats.sac.stlo, tr.stats.sac.stla
    # dist.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
    st_dist = tr.stats.sac.dist
    dist.append(st_dist)
    dt = tr.stats.delta
    n1 = int((t1-b)/dt); n2 = int((t2-b)/dt)
    d = tr.data[n1: n2+1]
    if np.abs(d).max() < 1e-8:
        continue
    d /= np.abs(d).max()
    ccf.append(d)

t = np.arange(len(d)) * dt + t1
dist = np.array(dist)
ccf = np.array(ccf)
index = np.argsort(dist)
ccf = ccf[index]
dist = dist[index]

plt.figure(figsize=(10, 7))
ax = plt.subplot(111)
ax.tick_params(labelsize=12)
for i in range(ccf.shape[0]):
    plt.plot(t, ccf[i] * scale + dist[i], lw=2, color='k')

plt.xlabel('Lag time (s)', fontsize=13)
plt.ylabel('Interstation distance (m)', fontsize=13)
plt.xlim(t1, t2)
plt.grid(ls=':', lw=1.5, color='#AAAAAA')
plt.tight_layout()
plt.show()
