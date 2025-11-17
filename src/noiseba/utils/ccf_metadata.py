"""CCF metadata management utilities."""

from __future__ import annotations
import sqlite3
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Callable

import numpy as np
import pandas as pd
import shutil
from obspy.core import AttribDict, Trace


class CCFIndex:
    def __init__(
        self,
        station_df: pd.DataFrame,
        pairs,
        *,
        build_ccf_index: Optional[Callable] = None,
        query_ccf_index: Optional[Callable] = None,
        save_ccf_index: Optional[Callable] = None,
        load_ccf_index: Optional[Callable] = None,
        front_k_pairs: Optional[Callable] = None,
    ) -> None:
        self._sta = station_df.reset_index(drop=True)
        self._pairs = pairs[:]  # deep copy
        self._query_ccf_index = query_ccf_index or _f.query_ccf_index
        self._build_ccf_index = build_ccf_index or _f.build_ccf_index
        self._front_k_pairs = front_k_pairs or _f.front_k_pairs
        self._save_ccf_index = save_ccf_index or _f.save_ccf_index
        self._load_ccf_index = load_ccf_index or _f.load_ccf_index

        self._ccf_index = None

    @classmethod
    def from_sta_file(
        cls,
        stainfo_path: str,
        *,
        k: Optional[int] = None,
        read_stations_func: Optional[Callable[[str], pd.DataFrame]] = None,
        front_k_pairs_func: Optional[Callable] = None,
        **kw,
    ) -> "CCFIndex":
        """
        Create a CCFIndex instance from a station file - pattern(sta_name, x, y).

        This factory method reads station information from a text file and 
        initializes a CCFIndex with the first k stations and their pairs.

        Parameters
        ----------
        stainfo_path : str
            Path to the station information file. The file should contain 
            whitespace-separated values with columns: station_name, x_coordinate, y_coordinate.
        k : int, optional
            Number of stations to consider from the beginning of the file. 
            If None, all stations in the file will be used. Default is None.
        read_stations : callable, optional
            Function to read station data from file. Must accept a file path 
            and return a DataFrame with 'sta', 'x', and 'y' columns. 
        front_k_pairs : callable, optional
            Function to generate pairs from the first k stations. 

        Returns
        -------
        CCFIndex
            A new CCFIndex instance initialized with stations and pairs 
            derived from the station file.

        Examples
        --------
        >>> ccf_index = CCFIndex.from_sta_file('stations.txt', k=10)
        
        >>> # Using custom functions
        >>> ccf_index = CCFIndex.from_sta_file(
        ...     'stations.txt', 
        ...     k=5, 
        ...     read_stations=custom_reader,
        ...     front_k_pairs=custom_pair_generator
        ... )
        """
        read_stations_func = read_stations_func or read_stations
        front_k_pairs_func = front_k_pairs_func or front_k_pairs

        station_df = read_stations_func(stainfo_path)
        k = len(station_df) if k is None else k
        pairs = front_k_pairs_func(station_df, k)

        return cls(station_df, pairs, 
                  front_k_pairs=front_k_pairs_func,
                  **kw)

    @classmethod
    def load_index(cls, ccf_path: str, load_ccf_index_func=None) -> pd.DataFrame:
        """
        Load CCF index from the specified path.
        """
        if not isinstance(ccf_path, str):
            raise ValueError("ccf_path must be a string")
        
        if not ccf_path.strip():
            raise ValueError("ccf_path cannot be empty")
        
        load_ccf_index_func = load_ccf_index_func or load_ccf_index
        try:
            return load_ccf_index_func(ccf_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CCF index file not found at path: {ccf_path}")
        except Exception as e:
            raise Exception(f"Failed to load CCF index from {ccf_path}: {str(e)}")


    @property
    def stations(self) -> pd.DataFrame:
        """Station info (coordinates, names, etc.), read-only"""
        return self._sta

    @property
    def pairs(self) -> List[Tuple[int, int, str, str]]:
        """List of station pairs (idx_a, idx_b, sta_a, sta_b), read-only"""
        return self._pairs

    @property
    def ccf_index(self) -> pd.DataFrame:
        """
        Built CCF metadata table (distance, path, quality scores, etc.).
        Raises error if not built yet.
        """
        if self._ccf_index is None:
            raise RuntimeError(
                "CCF index has not been built yet. Call .build_index() first."
            )
        return self._ccf_index

    def query_index(
        self,
        *,
        max_distance: Optional[float] = None,
        station_names: Optional[List[str]] = None,
        storage_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Filter the built ccf_index
        Example: ccf.query_index(max_distance=50000, station_names=["A001", "A005"])
        """
        if self._ccf_index is None:
            raise RuntimeError(
                "CCF index has not been built yet. Call .build_index() first."
            )

        return self._query_ccf_index(
            self._ccf_index,
            max_distance=max_distance,
            station_names=station_names,
            storage_type=storage_type,
            tags=tags,
        )

    def build_index(
        self,
        dt: float,
        nwin: int,
        ccf_root_path: str | Path,
        *,
        storage_type: str = "zarr",
        quality_scores: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        component: str = "Z",
        **extra_build_kwargs,
    ) -> "CCFIndex":
        """
        Core method: generate the CCF metadata table
        station_df and pairs are provided automatically by the object
        """
        if self._ccf_index is not None:
            raise RuntimeError("Index already built. Create a new instance to rebuild.")

        self._ccf_index = self._build_ccf_index(
            idx_pairs=self._pairs,
            station_df=self._sta,
            dt=dt,
            nwin=nwin,
            ccf_path=str(ccf_root_path),
            storage_type=storage_type,
            quality_scores=quality_scores,
            tags=tags,
            component=component,
            **extra_build_kwargs,
        )
        return self._ccf_index.copy()

    def save_index(self, path: str | Path, format: str = "csv"):
        """
        Save the built ccf_index to disk
        Supported formats: csv / parquet / sqlite
        """
        if self._ccf_index is None:
            raise RuntimeError("Nothing to save: index not built yet.")
        self._save_ccf_index(self._ccf_index, str(path), format=format)

  


# Reference to functions in this module for use in CCFIndex class
_f = sys.modules[__name__]


def read_stations(txt_path: str) -> pd.DataFrame:
    """
    Read station coordinates from text file.

    Parameters
    ----------
    txt_path : str
        Path to station file with format 'sta_name x y'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['sta', 'x', 'y'], indexed 0..N-1.
    """
    df = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["sta", "x", "y"])
    df["sta"] = df["sta"].astype(str)
    return df.reset_index(drop=True)


def front_k_pairs(df: pd.DataFrame, k: int = 5) -> List[Tuple[int, int, str, str]]:
    """
    Get first k station pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Station DataFrame.
    k : int, optional
        Number of stations to consider, by default 5.

    Returns
    -------
    List[Tuple[int, int, str, str]]
        List of tuples (i, j, sta_i, sta_j).
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

    Parameters
    ----------
    correlations : np.ndarray
        Correlation array of shape (n_pairs, n_lags).
    pairs : list
        List of station pairs.
    station_dataframe : pd.DataFrame
        DataFrame with station information.
    dt : float
        Time sampling interval.
    frequency_band : tuple, optional
        Optional frequency band for filtering.
    output_directory : str, optional
        Directory to write SAC files.
    component : str, optional
        Seismic component.
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
    """
    Build CCF index DataFrame.

    Parameters
    ----------
    idx_pairs : list
        List of station index pairs.
    station_df : pd.DataFrame
        Station information DataFrame.
    dt : float
        Time sampling interval.
    nwin : int
        Number of windows.
    ccf_path : str
        Path to CCF storage.
    storage_type : str, optional
        Storage type ('sac', 'zarr', etc.).
    quality_scores : list of float, optional
        Optional quality scores.
    tags : list of str, optional
        Optional tags.
    component : str, optional
        Component name.

    Returns
    -------
    pd.DataFrame
        DataFrame with CCF metadata.
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
            storage_path = str(
                Path(ccf_path)
                / f"CCF_{i:02d}_{j:02d}_{name_a}_{name_b}.{component}.sac"
            )
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


def save_ccf_index(
    df: pd.DataFrame, path: str, format: str = "csv", table_name: str = "ccf_records"
) -> None:
    """
    Save CCF index to file.

    Parameters
    ----------
    df : pd.DataFrame
        CCF index DataFrame.
    path : str
        Output file path.
    format : str, optional
        File format ('csv', 'parquet', 'sqlite').
    table_name : str, optional
        Table name for SQLite.
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


def load_ccf_index(
    path: str, format: Optional[str] = None, table_name: str = "ccf_records"
) -> pd.DataFrame:
    """
    Load CCF index from file.

    Parameters
    ----------
    path : str
        Input file path.
    format : str, optional
        File format (auto-detected if None).
    table_name : str, optional
        Table name for SQLite.

    Returns
    -------
    pd.DataFrame
        CCF index DataFrame.
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
    station_indices=None,
    station_names=None,
    max_distance=None,
    station_pairs=None,
    tags=None,
    storage_type=None,
) -> pd.DataFrame:
    """
    Query CCF index DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        CCF index DataFrame.
    station_indices : list or set, optional
        Filter by station indices.
    station_names : list or set, optional
        Filter by station names.
    max_distance : float, optional
        Maximum distance filter.
    station_pairs : list of tuple, optional
        Filter by specific station pairs.
    tags : str or list, optional
        Filter by tags.
    storage_type : str, optional
        Filter by storage type.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
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
        mask &= (df["idx_a"].astype(str) + "," + df["idx_b"].astype(str)).isin(
            [f"{i},{j}" for i, j in station_pairs_set]
        )

    return df[mask].reset_index(drop=True)
