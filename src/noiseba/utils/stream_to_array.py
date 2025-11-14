from typing import Tuple, List
import numpy as np
from obspy.core import Stream


def stream_to_array(stream: Stream) -> Tuple[np.ndarray, List[str]]:
    """
    Convert ObsPy Stream to NumPy array with station names.

    Args:
        stream: ObsPy Stream containing seismic data

    Returns:
        Tuple of (data_array, station_names)
        data_array: (n_stations, n_samples) shaped array
        station_names: List of station identifiers
    """
    if not stream:
        raise ValueError("Empty stream!")

    stream.detrend("demean")
    stream.detrend("linear")

    # Check sampling rate consistency
    reference_sr = stream[0].stats.sampling_rate
    mismatched_stations = [tr.stats.station for tr in stream if tr.stats.sampling_rate != reference_sr]
    if mismatched_stations:
        raise ValueError(f"Sampling rate mismatch! Stations: {','.join(mismatched_stations)}")

    # Check data length consistency
    reference_npts = stream[0].stats.npts
    mismatched_lengths = [tr.stats.station for tr in stream if tr.stats.npts != reference_npts]
    if mismatched_lengths:
        raise ValueError(f"Length mismatch! Stations: {','.join(mismatched_lengths)}")

    # Stack trace data: (n_stations, n_samples)
    data = np.stack([trace.data for trace in stream], axis=0, dtype=np.float32)
    stations = [trace.stats.station for trace in stream]
    return data, stations
