"""CCF metadata management utilities."""

import sqlite3
from pathlib import Path
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import shutil
from obspy.core import AttribDict, Trace


def read_stations(txt_path: str) -> pd.DataFrame:
    """Read station coordinates from text file.

    Args:
        txt_path: Path to station file with format 'sta_name x y'

    Returns:
        DataFrame with columns ['sta', 'x', 'y'], indexed 0..N-1
    """
    df = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["sta", "x", "y"])
    df["sta"] = df["sta"].astype(str)
    return df.reset_index(drop=True)


def front_k_pairs(df: pd.DataFrame, k: int = 5) -> List[Tuple[int, int, str, str]]:
    """Get first k station pairs.

    Args:
        df: Station DataFrame
        k: Number of stations to consider

    Returns:
        List of tuples (i, j, sta_i, sta_j)
    """
    pairs = []
    n = len(df)
    for i in range(k - 1):
        for j in range(i + 1, min(k, n)):
            pairs.append((i, j, df.at[i, "sta"], df.at[j, "sta"]))
    return pairs


def write_front_k_sac(
    correlations: np.ndarray,
    pairs,
    station_dataframe: pd.DataFrame,
    dt: float,
    frequency_band: tuple | None = None,
    output_directory: str = "./sac_front_k",
    component: str = "Z",
):
    """
    Write first k cross-correlations to SAC files.

    Args:
        correlations: Correlation array of shape (n_pairs, n_lags)
        pairs: List of station pairs
        station_dataframe: DataFrame with station information
        dt: Time sampling interval
        frequency_band: Optional frequency band for filtering
        output_directory: Directory to write SAC files
        component: Seismic component
    """
    output_path = Path(output_directory)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for row, (i, j, sta_i, sta_j) in enumerate(pairs):
        trace = Trace(data=correlations[row].real)
        trace.stats.delta = dt
        trace.stats.sac = AttribDict(
            {
                "dist": np.hypot(
                    station_dataframe.at[i, "x"] - station_dataframe.at[j, "x"],
                    station_dataframe.at[i, "y"] - station_dataframe.at[j, "y"],
                ),
                "evlo": station_dataframe.at[i, "x"],
                "evla": station_dataframe.at[i, "y"],
                "stlo": station_dataframe.at[j, "x"],
                "stla": station_dataframe.at[j, "y"],
                "kstnm": sta_i,
                "kevnm": f"{sta_i}-{sta_j}",
            }
        )

        # Apply filter if specified
        if frequency_band is not None:
            fmin, fmax = frequency_band
            if fmin >= fmax or fmin <= 0:
                raise ValueError(f"Invalid freq_band: {frequency_band}")
            trace.filter("bandpass", freqmin=fmin, freqmax=fmax)
            trace.stats.sac.user0 = fmin
            trace.stats.sac.user1 = fmax

        filename = f"CCF_{i:02d}_{j:02d}_{sta_i}_{sta_j}.{component}.sac"
        # Apply filter if specified
        if frequency_band is not None:
            fmin, fmax = frequency_band
            if fmin >= fmax or fmin <= 0:
                raise ValueError(f"Invalid freq_band: {frequency_band}")
            trace.filter("bandpass", freqmin=fmin, freqmax=fmax)
            trace.stats.sac.user0 = fmin
            trace.stats.sac.user1 = fmax

        filename = f"CCF_{i:02d}_{j:02d}_{sta_i}_{sta_j}.{component}.sac"
        trace.write(str(output_path / filename), format="SAC")


def build_ccf_index(
    idx_pairs,
    station_df: pd.DataFrame,
    dt: float,
    nwin: int,
    ccf_path: str,
    storage_type: str = "zarr",
    quality_scores: Optional[List[float]] = None,
    tags: Optional[List[str]] = None,
    component: str = "Z",
) -> pd.DataFrame:
    """Build CCF index DataFrame.

    Args:
        idx_pairs: List of station index pairs
        station_df: Station information DataFrame
        dt: Time sampling interval
        nwin: Number of windows
        ccf_path: Path to CCF storage
        storage_type: Storage type ('sac', 'zarr', etc.)
        quality_scores: Optional quality scores
        tags: Optional tags
        component: Component name

    Returns:
        DataFrame with CCF metadata
    """
    idx_to_name = dict(enumerate(station_df["sta"]))
    idx_to_coord = dict(enumerate(zip(station_df["x"], station_df["y"])))

    records = []

    for k, (i, j, *_) in enumerate(idx_pairs):
        name_a = idx_to_name[i]
        name_b = idx_to_name[j]
        x_a, y_a = idx_to_coord[i]
        x_b, y_b = idx_to_coord[j]
        dist = float(np.hypot(x_b - x_a, y_b - y_a))

        if storage_type == "sac":
            storage_path = str(Path(ccf_path) / f"CCF_{i:02d}_{j:02d}_{name_a}_{name_b}.{component}.sac")
        elif storage_type == "zarr":
            storage_path = f"{i:02d}_{j:02d}_{name_a}_{name_b}"  # group key
        else:
            storage_path = None

        record = {
            "idx_a": i,
            "idx_b": j,
            "sta_a": name_a,
            "sta_b": name_b,
            "x_a": x_a,
            "y_a": y_a,
            "x_b": x_b,
            "y_b": y_b,
            "distance": dist,
            "dt": dt,
            "nwin": nwin,
            "storage_type": storage_type,
            "storage_path": storage_path,
            "quality_score": quality_scores[k] if quality_scores else None,
            "tags": tags[k] if tags else None,
        }
        records.append(record)

    return pd.DataFrame(records)


def save_ccf_index(df: pd.DataFrame, path: str, format: str = "csv", table_name: str = "ccf_records") -> None:
    """Save CCF index to file.

    Args:
        df: CCF index DataFrame
        path: Output file path
        format: File format ('csv', 'parquet', 'sqlite')
        table_name: Table name for SQLite
    """
    format = format.lower()
    if format == "csv":
        if not path.endswith(".csv"):
            path += ".csv"
        df.to_csv(path, index=False)
        print(f"Saved CCF index to CSV: {path}")

    elif format == "parquet":
        if not path.endswith(".parquet"):
            path += ".parquet"
        df.to_parquet(path, index=False)
        print(f"Saved CCF index to Parquet: {path}")

    elif format == "sqlite":
        if not path.endswith(".sqlite"):
            path += ".sqlite"
        conn = sqlite3.connect(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        print(f"Saved CCF index to SQLite: {path} (table: {table_name})")

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_ccf_index(path: str, format: Optional[str] = None, table_name: str = "ccf_records") -> pd.DataFrame:
    """Load CCF index from file.

    Args:
        path: Input file path
        format: File format (auto-detected if None)
        table_name: Table name for SQLite

    Returns:
        CCF index DataFrame
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if format is None:
        ext = path_obj.suffix.lower()
        if ext == ".csv":
            format = "csv"
        elif ext == ".parquet":
            format = "parquet"
        elif ext in [".sqlite", ".db"]:
            format = "sqlite"
        else:
            raise ValueError(f"Invalid format extension: {ext}")

    format = format.lower()
    if format == "csv":
        df = pd.read_csv(path)
    elif format == "parquet":
        df = pd.read_parquet(path)
    elif format == "sqlite":
        conn = sqlite3.connect(path)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Loaded CCF index from {path} ({format}), {len(df)} records.")
    return df


def query_ccf_index(
    df: pd.DataFrame,
    station_indices = None,
    station_names   = None,
    max_distance    = None,
    station_pairs   = None,
    tags            = None,
    storage_type    = None,
) -> pd.DataFrame:
    """Query CCF index DataFrame.

    Args:
        df: CCF index DataFrame
        station_indices: Filter by station indices
        station_names: Filter by station names
        max_distance: Maximum distance filter
        station_pairs: Filter by specific station pairs
        tags: Filter by tags
        storage_type: Filter by storage type

    Returns:
        Filtered DataFrame
    """
    mask = pd.Series(True, index=df.index)

    if station_indices is not None:
        station_indices = set(station_indices)
        mask &= df["idx_a"].isin(station_indices) & df["idx_b"].isin(station_indices)

    if station_names is not None:
        station_names = set(station_names)
        mask &= df["sta_a"].isin(station_names) & df["sta_b"].isin(station_names)

    if max_distance is not None:
        mask &= df["distance"] <= max_distance

    if tags is not None:
        if isinstance(tags, str):
            tags = [tags]
        mask &= df["tags"].isin(tags)

    if storage_type is not None:
        mask &= df["storage_type"] == storage_type

    if station_pairs is not None:
        station_pairs_set = set(station_pairs)
        mask &= (df["idx_a"].astype(str) + "," + df["idx_b"].astype(str)).isin([f"{i},{j}" for i, j in station_pairs_set])

    return df[mask].reset_index(drop=True)
