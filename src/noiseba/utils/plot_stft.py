import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import stft
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
from datetime import timedelta

def plot_stft(stream, win_sec=5.0, overlap=0.5, china=False):
    """ Time-frequency plot of a stream using STFT

    Args:
        stream : obspy.stream.Stream
        win_sec (float, optional): minimal window size in sec. Defaults to 2.0.
        overlap (float, optional): window overlap. Defaults to 0.5.
        china (bool, optional): whether to use chinese time zone. Defaults to False.
    """
    tr = stream[0]
    data = tr.data.astype(np.float64)
    fs = tr.stats.sampling_rate

    nperseg = int(win_sec * fs) # points
    noverlap = int(nperseg * overlap)
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')

    if china:
        start_time = tr.stats.starttime.datetime + timedelta(hours=8)
    else:
        start_time = tr.stats.starttime.datetime

    times_sec = [start_time + timedelta(seconds=s) for s in tr.times()]
    t_sec = [start_time + timedelta(seconds=s) for s in t]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 4], "hspace": 0.05})

    ax1.plot(times_sec, data, color='black', linewidth=0.8)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', which='both', bottom=False)

    abs_Zxx = np.abs(Zxx)
    norm = LogNorm(vmin=1, vmax=30)
    im = ax2.pcolormesh(t_sec, f, np.abs(Zxx), shading='auto', cmap='plasma', norm=norm)
    ax2.set_ylabel("Freq (Hz)", fontsize=11)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(base=10.0))

    fig.autofmt_xdate() # rotate

    # colorbar
    cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', shrink=0.35, pad=0.35)
    cbar.set_label("Amplitude", fontsize=10)
    cbar.ax.tick_params(which='minor', bottom=False)
    # plt.show()
    fig.savefig('STFT.png', dpi=300, bbox_inches='tight')
    # return t, t_sec

if __name__ == '__main__':
    
    from obspy import read
    file = r'/media/wdp/disk4/site1_line1/reef/pre_4.13/451027849.Z.sac'
    st = read(file, format='SAC')
    plot_stft(st, win_sec=20, china=True)