import numpy as np
import logging

from obspy import read, Trace, Stream
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm

logging.basicConfig(filename='1-preprocess.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    encoding='utf-8')

def smart_decimate(trace: Trace, target_rate: Optional[float]) -> Trace:
    original_rate = trace.stats.sampling_rate
    if original_rate <= target_rate or target_rate == None:
        return trace

    factor = int(np.ceil(original_rate / target_rate))
    nyquist = target_rate / 2.0
    trace.filter("lowpass", freq=nyquist, corners=4, zerophase=True)
    trace.decimate(factor=factor, no_filter=True)
    return trace

def process_single_trace(
                tr: Trace, 
                output_dir: Path,
                target_rate: Optional[float] = None, 
                freq_min: Optional[float] = None, 
                freq_max: Optional[float] = None) -> str:
    
    """process single Trace and save"""
    try:
        tr.detrend("demean")
        tr.detrend("linear")

        if freq_min is not None and freq_max is not None:
            freq_max = min(tr.stats.sampling_rate // 2, freq_max)
            freq_min = max(freq_min, 1. / tr.times()[-1])
            tr.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)

        elif freq_min is not None:
            freq_min = max(freq_min, 1. / tr.times()[-1])
            tr.filter("highpass", freq=freq_min, corners=4, zerophase=True)

        elif freq_max is not None:
            freq_max = min(tr.stats.sampling_rate, freq_max)
            tr.filter("lowpass", freq=freq_max, corners=4, zerophase=True)

        else:
            logging.warning(f"Skipping filter for {tr.id}: no frequency limits provided")

        if target_rate is not None:
            tr = smart_decimate(tr, target_rate)

        station_name = f"45{tr.stats.station}"
        channel = tr.stats.channel[-1]
        output_path = output_dir / f"{station_name}.{channel}.sac"

        if output_path.exists():
            output_path.unlink()

        tr.write(output_path.as_posix(), format="SAC")
        return f"{station_name} ✅"
    except Exception as e:
        logging.error(f"Failed to process {tr.stats.station}: {e}")
        return f"{tr.stats.station} ❌ {e}"
    

def batch_process(input_dir: Path, 
                  output_dir: Path, 
                  target_rate: float = 200.0,
                  freq_min: Optional[float] = None,
                  freq_max: Optional[float] = None,
                  start_time: Optional[float] = None, 
                  end_time: Optional[float] = None, 
                  max_workers: int = 8,
                  component: str = "Z"):
    """Batch process SAC files in input directory"""

    logging.info(f"Starting data processing")
    logging.info(f"Input directory: {input_dir.as_posix()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stream = Stream()
    print("Reading...")
    logging.info("Reading SAC files...")

    for sac_file in input_dir.glob("*{0}.sac".format(component)):
        try:
            stream += read(sac_file)
        except Exception as e:
            logging.error(f"Import Failed: {sac_file.name}: {e}")
            print(f"Import Failed: {sac_file.name}: {e}")
   
    # absord sampling_rate deviate >0.01 Hz
    rates = np.array([tr.stats.sampling_rate for tr in stream])
    avg_rate = np.mean(rates)
    skipped_st = []
    for idx in np.where(abs(rates - avg_rate) > 0.01)[0][::-1]:
        skipped_st.append('45' + stream[idx].stats.station)
        del stream[idx]
        logging.warning(f"Removing {stream[idx].stats.station} trace with sampling_rate deviate >0.01 Hz")
        print(f'remove {stream[idx].stats.station} trace with sampling_rate deviate >0.01 Hz')
    
    # merge and trim 
    stream.merge(method=1, fill_value=0)
    stream.trim(start_time, end_time, pad=True, fill_value=0)
    
    text = f"---------------Processing {len(stream)} Trace-----------------"
    logging.info(text)
    print(text)
    print('\n')

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_trace, tr, output_dir, target_rate, freq_min, freq_max)
            for tr in stream
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing traces"):
            results.append(future.result())

    print('\n')
    print("-------------------------Done-----------------------------------")
    logging.info("Batch processing completed")

if __name__ == "__main__":

    import datetime
    st = read(r'/media/wdp/disk4/site1_line1/gate1/453000036.0001.N.sac') # choice one tos access meta

    start_time = st[0].stats.starttime + datetime.timedelta(minutes=5)
    # end_time = start_time + datetime.timedelta(minutes=5)
    end_time = st[0].stats.endtime

    config = {
        "input_dir": Path("/media/wdp/disk4/site1_line1/gate1"),
        "output_dir": Path("/media/wdp/disk4/site1_line1/gate1/pre_gate_data"),
        "target_rate": 100,     # decimate rate
        "freq_min": 0.1,          # lower filter
        "freq_max": 45,           # upper filter
        "start_time": start_time, # cut begin
        "end_time": end_time,     # cut end
        "max_workers": 36         # used cpu cores
    }

    batch_process(**config)