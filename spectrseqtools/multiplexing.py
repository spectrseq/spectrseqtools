import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from dbscan1d.core import DBSCAN1D
from sklearn.metrics import silhouette_score
from typing import List

from lionelmssq.masses import EXPLANATION_MASSES
from lionelmssq.deconvolution import PREPROCESS_TOL, MIN_MS1_CHARGE_STATE

rt = get_mono()

@dataclass
class PrecursorPeak:
    scan_time: float
    mz: float
    intensity: float
    charge: int

COL_TYPES_PRECURSOR = {
    "scan_time": pl.Float64,
    "mz": pl.Float64,
    "intensity": pl.Float64,
    "charge": pl.Int64,
}


def initialize_raw_file_iterator_ungrouped(
    file_path: str,
) -> ms_ditp.data_source.thermo_raw_net.ThermoRawLoader:
    """
    Initialize iterator over scans in ThermoFisher RAW file format.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.

    Returns
    -------
    raw_file : ms_deisotope.data_source.thermo_raw_net.ThermoRawLoader
        Iterator over scans from RAW file.

    """
    # Read data from file
    raw_file = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(
        file_path, _load_metadata=True
    )

    return raw_file

def get_precursor_peaks(raw_file):

    scan_processor = ms_ditp.ScanProcessor(raw_file)

    precursor_scans = []

    while True:
        try:
            scan_bunch = next(raw_file)

            ms1, precursors, _ = scan_processor.process_scan_group(scan_bunch.precursor, scan_bunch.products)

            scan_time = ms1.scan_time
            for p in precursors:
                precursor_scans.append(PrecursorPeak(
                    scan_time = scan_time,
                    mz = p.peak.mz,
                    intensity = p.peak.intensity,
                    charge = 0 if type(p.charge) is not int else p.charge,
                ))
        except StopIteration:
            break
    
    return precursor_scans

def get_representative_peaks(raw_file):

    precursor_scans = get_precursor_peaks(raw_file)

    df_precursors = pl.DataFrame(
        data=np.array(
            [[peak.__dict__[key] for key in COL_TYPES_PRECURSOR.keys()] for peak in precursor_scans]
        ),
        schema=COL_TYPES_PRECURSOR,
    )

    df_precursors = df_precursors.filter(pl.col("charge")>MIN_MS1_CHARGE_STATE).sort("mz").with_columns(
            (
                abs(pl.col("mz").shift(1) - pl.col("mz"))
                / pl.col("mz").shift(1)
            )
            .fill_null(0)
            .fill_nan(0)
            .gt(PREPROCESS_TOL)
            .cum_sum()
            .alias("cluster_group"))

    df_precursors_agg = df_precursors.group_by("cluster_group").agg(average_mz = pl.col("mz").mean(),
                                                                    intensity = pl.col("intensity").sum(),
                                                                    max_charge = pl.col("charge").max()).sort("cluster_group")
    return df_precursors_agg

def calculate_cluster_score(scan_times: pl.Series) -> float:
    """
    Determine score measuring how clustered each scan peaks is.

    By scan time, use DBSCAN and Silhouette score to evaluate peak clustering.

    Parameters
    ----------
    scan_times : polars.Series
        Scan times.

    Returns
    -------
    score : float
        Silhouette score for the DBSCAN cluster of scan times.
    """
    # Transform series to numpy array
    scan_times = scan_times.sort().to_numpy()

    # Cluster scan times using 1D DBSCAN
    clusters = DBSCAN1D(eps=0.5, min_samples=10).fit_predict(scan_times)

    # Flatten array containing scan times
    scan_times = scan_times.reshape(-1, 1)

    # Raise error if no cluster was found
    if len(set(clusters)) == 0:
        raise NotImplementedError("No cluster was found. This should not be possible.")

    # Return silhouette score if multiple clusters were found
    if len(set(clusters)) > 1:
        return silhouette_score(scan_times, clusters)

    # Return minimum score if only noise was found, i.e. cluster == -1
    if list(set(clusters))[0] == -1:
        return -1.0

    # Return neutral score if only one (non-noisy) cluster was found
    return 0.0
