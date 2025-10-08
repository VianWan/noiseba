import numpy as np
from obspy import read, UTCDateTime
from numpy.fft import fft, ifft, ifftshift
from numpy import ones, convolve
import re
import glob
import os
from obspy.core import Trace, AttribDict
from scipy.signal import hilbert, sosfilt, zpk2sos, sosfilt, iirfilter
from scipy.signal.windows import hann
import matplotlib.pyplot as plt

from obspy.signal.cross_correlation import correlate

def process_trace(data, nfft, operator, p=0.01):
    d = data - np.mean(data);
    # d = np.sign(d)
    # plt.plot(d)
    # plt.show()
    n = len(d)
    sn = int(n*p); hn = 2 * sn; h = hann(hn)
    d[:sn] *= h[:sn]; d[-sn:] *= h[-sn:]
    fd = fft(d, nfft)
    fd = fd / np.convolve(abs(fd), operator, 'same')
    fd[np.isnan(fd)] = 0
    # fd = np.real(ifft(fd))
    return fd

sta = {}
with open('/media/wdp/disk4/site1_line1/xy.txt', 'r') as fin:
    for line in fin.readlines():
        tmp = line.strip().split()
        sta[tmp[1]] = [float(tmp[0]), 0] # station name, x, y

f1 = 0.1; f2 = 45
dt = 0.01; fs = 1 / dt
h = 10; N = 2 * h + 1
operator = np.ones(N) / N
segt = 0.5 * 120
ot = 0.25 * 120; pt = segt - ot
segn = int(segt/dt)
pn = int((segt-ot)/dt)
nfft = segn #2 * int(segt*0.55/dt) + 1
t = np.arange(nfft) * dt - nfft//2*dt
t1 = -25; t2 = 25
tn1 = int((t1-t.min())/dt); tn2 = int((t2-t.min())/dt) + 1
t = t[tn1: tn2]
cor = Trace(t)
cor.stats.delta = dt
cor.stats.sac = AttribDict({})
cor.stats.sac.b = t[0]
cor.data = np.zeros(nfft, dtype=float)

print('Read data ...')
try:
    st = read('/media/wdp/disk4/site1_line1/pre_data/*.sac')
    lm = len(st)
    if lm < 2:
        print('No enough data!!!')
        exit(1)
except Exception:
    pass

st.detrend('demean')
st.detrend('linear')

ccfdir = 'CCF_ZZ/'
if not os.path.exists(ccfdir):
    os.mkdir(ccfdir)
utc1 = st[0].stats.starttime
utc2 = st[0].stats.endtime

for tr in st:
    if tr.stats.starttime < utc1:
        utc1 = tr.stats.starttime
    if tr.stats.endtime > utc2:
        utc2 = tr.stats.endtime

ln = int((utc2-utc1+segt)/dt)
print('Store data ...')
ns = []; all_data = np.zeros((lm, ln))
for i in range(lm):
    n1 = int((st[i].stats.starttime-utc1)/dt)
    all_data[i, n1: n1+len(st[i].data)] = st[i].data
    k = '45' + st[i].stats.station
    ns.append(k)
print('Preprocess data ...')
data_fd = []
for i in range(lm):
    dn1 = 0
    tmp = []
    while dn1 <= ln-segn:
        dn2 = dn1 + segn
        fd = process_trace(all_data[i, dn1: dn2], nfft, operator)
        tmp.append(fd)
        dn1 += pn
    data_fd.append(tmp)

print('CCF ...')
nstack = len(data_fd[0])
for si in range(lm-1):
    xi, yi = sta[ns[si]][:2]
    for sj in range(si+1, lm):
        xj, yj = sta[ns[sj]][:2]
        ccfname = 'COR_' + ns[si] +'_' + ns[sj] + '.sac'
        ccfd = np.zeros(nfft, dtype=complex)
        dist = ( (xi-xj)**2 + (yi-yj)**2 )**0.5
        cor.stats.sac.dist = dist
        for sk in range(nstack):
            ccfd += (data_fd[si][sk]*np.conjugate(data_fd[sj][sk]))
            # ccfd = np.correlate(data_fd[si][sk], data_fd[sj][sk],'full')
            # ccfd = correlate(data_fd[si][sk], data_fd[sj][sk], segn-1)

        cor.data = ifftshift(ifft(ccfd)).real[tn1: tn2]
        # cor.data = ccfd[tn1: tn2]
        # cor.data = ccfd
        cor.stats.sac.evlo = xi; cor.stats.sac.evla = yi
        cor.stats.sac.stlo = xj; cor.stats.sac.stla = yj
        cor.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=True)
        cor.write(ccfdir+ccfname)    
