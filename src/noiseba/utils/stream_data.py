import numpy as np

from typing import List, Tuple

def stream_data(stream) -> Tuple[np.ndarray, List[str]]:
    """Convert stream to array and trim data according to the dataset's minimal and maximum time."""
    stream.detrend('demean')
    stream.detrend('linear')
    # stream.taper(max_percentage=0.05, type='cosine')
    begin_time = np.max([i.stats.starttime for i in stream])
    end_time = np.min([i.stats.endtime for i in stream])
    stream.trim(begin_time, end_time)
    data = np.array([tr.data for tr in stream])
    station = ['45' + tr.stats.station for tr in stream]    

    return data, station