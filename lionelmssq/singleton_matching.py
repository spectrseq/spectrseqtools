import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from dbscan1d.core import DBSCAN1D
from sklearn.metrics import silhouette_score
from typing import List

from lionelmssq.common import initialize_raw_file_iterator
from lionelmssq.masses import MZ_MASSES

rt = get_mono()

PPM_TOLERANCE = 10
THEORETICAL_BOUNDARY_FACTOR = 2
COL_TYPES_RAW = {
    "scan_id": pl.Int32,
    "scan_time": pl.Float64,
    "peak_idx": pl.Int64,
    "intensity": pl.Float64,
    "mz": pl.Float64,
}


def extract_mz_data(file_path: str) -> pl.DataFrame:
    """
    Extract m/z data in MS2 scans from ThermoFisher RAW files.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.

    Returns
    -------
    df_mz : polars.DataFrame
        Dataframe containing mass-to-charge ratios and intensities.

    """
    # Initialize the first scan from an iterator of the RAW file
    raw_file_read = initialize_raw_file_iterator(file_path=file_path)
    bunch = next(raw_file_read)

    peak_list = []
    for _ in tqdm.tqdm(
        range(len(raw_file_read) - 1), desc="Extract m/z data from MS2 scans"
    ):
        # Only consider MS2 scans
        if bunch.ms_level == 2:
            # Centroid the scans
            bunch.pick_peaks()

            # Extract and generate dataframe of m/z for each scan (without deisotoping)
            peak_list += process_scan(bunch)

        # Move on to the next scan
        bunch = next(raw_file_read)

    # Build dataframe from peak list
    df_mz = pl.DataFrame(
        data=np.array(
            [[peak.__dict__[key] for key in COL_TYPES_RAW.keys()] for peak in peak_list]
        ),
        schema=COL_TYPES_RAW,
    )

    df_mz = df_mz.sort("mz").with_columns(
        (1e6 * abs(pl.col("mz").shift(1) - pl.col("mz")) / pl.col("mz").shift(1))
        .fill_null(0)
        .fill_nan(0)
        .gt(PPM_TOLERANCE)
        .cum_sum()
        .alias("ppm_group")
    )

    return df_mz


@dataclass
class RawPeak:
    scan_id: int
    scan_time: float
    peak_idx: int
    intensity: float
    mz: float


def process_scan(bunch: ms_ditp.data_source.Scan) -> List[RawPeak]:
    """
    Extract raw peaks from MS2 scan.

    Parameters
    ----------
    bunch : ms_deisotope.data_source.Scan
        ThermoFisher scan.

    Returns
    -------
    peak_list : List[RawPeak]
        List containing raw peak data.

    """
    # Return None if scan does not contain any peaks
    if len(bunch.peaks) <= 0:
        return []

    # Obtain scan time and scan ID
    scan_time = bunch.scan_time
    scan_id = int(bunch.scan_id.split("scan=")[-1])

    # Calculate theoretical bounds, i.e. accepted m/z range
    min_mz = MZ_MASSES["theoretical_mz"].min() * (
        1 - THEORETICAL_BOUNDARY_FACTOR * PPM_TOLERANCE / 1e6
    )
    max_mz = MZ_MASSES["theoretical_mz"].max() * (
        1 + THEORETICAL_BOUNDARY_FACTOR * PPM_TOLERANCE / 1e6
    )

    peak_list = []
    for idx in range(len(bunch.peaks)):
        mz = bunch.peaks[idx].mz

        # Only consider peaks with mass within theoretical bounds
        if min_mz <= mz <= max_mz:
            peak_list.append(
                RawPeak(
                    scan_id=scan_id,
                    scan_time=scan_time,
                    peak_idx=idx,
                    intensity=bunch.peaks[idx].intensity,
                    mz=mz,
                )
            )
    return peak_list


def cluster_score(scantimes):
    """
    Provide a score on how clustered each of the scan peaks by scan time, using DBSCAN and Silhouette scores

    Parameters
    ----------
    scantimes : polars Series
        Series containing scan times

    Returns
    -------
    score : float
        Silhouette score for the DBSCAN cluster of scan times
    """

    scan_arr = scantimes.sort().to_numpy()

    # Cluster scan times using 1D DBSCAN
    dbs = DBSCAN1D(eps=0.5, min_samples=10)
    clusters = dbs.fit_predict(scan_arr)

    X = scan_arr.reshape(-1, 1)
    if len(set(clusters)) == 1:
        # If there are only noise clusters (cluster = -1), set score to -1.0
        if list(set(clusters))[0] == -1:
            score = -1.0
        # If there is only one cluster, set score to 0.0
        else:
            score = 0.0
    else:
        # Compute silhouette score otherwise
        score = silhouette_score(X, clusters)

    return score


def match_singletons(file_path: str) -> pl.DataFrame:
    """
    Match observed m/z from RAW file to theoretical m/z from reference mass table.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.

    Returns
    -------
    df_matches : polars.DataFrame
        Dataframe containing candidate nucleosides obtained by matching singletons.
    """
    df_mz = extract_mz_data(file_path)

    # Match m/z to theoretical m/z from the reference table
    df_mz = df_mz.sort("mz").join_asof(
        MZ_MASSES.sort("theoretical_mz"),
        left_on="mz",
        right_on="theoretical_mz",
        strategy="nearest",
    )

    # Compute mass error between observed and theoretical m/z
    df_mz = (
        df_mz.sort("mz")
        .with_columns(
            (1e6 * abs(pl.col("mz") - pl.col("theoretical_mz")) / pl.col("mz"))
            .fill_null(0)
            .fill_nan(0)
            .lt(PPM_TOLERANCE)
            .alias("is_match")
        )
        .filter(pl.col("is_match"))
        .sort(["nucleoside", "scan_time"])
    )

    df_mz_agg = df_mz.group_by("nucleoside").map_groups(
        lambda x: pl.DataFrame(
            {
                "nucleoside": x["nucleoside"][0],
                "cluster_score": cluster_score(x["scan_time"]),
                "count": len(x["nucleoside"]),
            }
        )
    )

    # Take mass error that is less than PPM_TOLERANCE and flatten list of candidate singletons
    df_matches = (
        df_mz_agg.filter(pl.col("cluster_score") >= 0)
        .select(["nucleoside", "count", "cluster_score"])
        .sort("count", descending=True)
        # .explode("nucleoside")
    )

    return df_matches
