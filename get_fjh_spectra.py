import numpy as np
from obspy import read
import sys
import ccfj
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import butter, sosfilt

argc = len(sys.argv)
if argc != 2:
    print('Usage python %s ccf_dir sub_array_file'%sys.argv[0])
    exit(1)

stack = sys.argv[1]
files = sorted(glob(stack+'/COR*.sac'))

f1 = 0; f2 = 40
t0 = 10
t1 = -t0; t2 = t0
print(f1, f2, t1, t2)
dist = []; data = []
ccfr = []; ccfi = []
index = 0
for sac in files:
    tr = read(sac)[0]
    y1 = tr.stats.sac.evla; x1 = tr.stats.sac.evlo
    y2 = tr.stats.sac.stla; x2 = tr.stats.sac.stlo
    gcarc = ( (x1-x2)**2 + (y1-y2)**2 ) ** 0.5
    if gcarc < 1e-6:
        continue
    dist.append(gcarc)
    b = tr.stats.sac.b
    dt = tr.stats.delta
    n1 = int((t1-b)/dt); n2 = int((t2-b)/dt)
    d = tr.data[n1: n2+1]
    d = np.fft.fftshift(d)
    tmp = np.fft.fft(d)
    n = len(tmp)
    ccfr.append(tmp.real[:n//2])
    index += 1

print('CCF files:', index)
ccfr = np.array(ccfr)
nd = len(d)
dist = np.array(dist)
index = np.argsort(dist)

dist = dist[index]
ccfr = ccfr[index]

print('rmin: %.6f rmax: %.6f'%(dist.min(), dist.max()))
c1 = 100; c2 = 1000; nc = 301
c = np.linspace(c1, c2, nc)
f = np.arange(nd) / dt / (nd-1)
fn1 = int(f1*nd*dt); fn2 = int(f2*nd*dt)
nf = fn2 - fn1 + 1; nr = len(dist)

im = ccfj.fj_noise(ccfr[:, fn1: fn2+1], dist, c,
                   f[fn1: fn2+1], fstride=1, itype=1,
                   func=1, num=36)

im[im<=0] = 0
for i in range(len(im[0])):
    if im[:, i].max() < 1e-8:
        continue
    im[:, i] /= np.abs(im[:, i]).max()
im = im**2
f = f[fn1: fn2+1]

plt.figure(figsize=(13, 6))
plt.pcolormesh(f, c/1e3, im, cmap='jet')
plt.plot(f, f*dist[-1]/1e3, lw=2, ls='--', color='w')
plt.ylim(c[0]/1e3, c[-1]/1e3)
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Phase velocity (km/s)', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.show()
