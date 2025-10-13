import importlib.resources
import ms_deisotope as ms_ditp
import numpy as np
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from dataclasses import dataclass
from typing import List, Tuple

from lionelmssq.common import initialize_raw_file_iterator

rt = get_mono()

PPM_TOLERANCE = 10
# TODO: Estimate the default value from sequence length (IMP: Should be
#  large enough to cover the charge states of the precursors!
DEFAULT_CHARGE_VALUE = 30
ISOTOPIC_SHIFT_FACTOR = 10
COL_TYPES_DEISOTOPED = {
    "scan_id": pl.Int32,
    "scan_time": pl.Float64,
    "peak_idx": pl.Int64,
    "intensity": pl.Float64,
    "neutral_mass": pl.Float64,
    "is_precursor": pl.Boolean,
    "is_deisotoped": pl.Boolean,
    "mz": pl.Float64,
}


# METHOD: To deconvolute/deisotope (which we use interchangeable because both
# happen at the same time) a MS2 scan, we determine all its peaks with
# the ms_deisotope package. We then identify the precursor peak (if it exists)
# and perform a check whether it has been deisotoped (since this is the only
# one that we can reliably obtain as there exists a reference m/z).
# We call a precursor correctly deisotoped if the m/z of the peak is:
# (1) less than the m/z of the precursor, and
# (2) greater than its isotopic shift times a given factor (here: 10).


class DeconvolutionParameters:
    def __init__(self, params: dict):
        # Set possibly bunch-dependent parameters (if given)
        self.charge_range = params.pop("charge_range", None)
        self.minimum_intensity = params.pop("minimum_intensity", None)

        # Set bunch-independent parameters
        self.bunch_independent_params = set_bunch_independent_params(params=params)


def set_bunch_independent_params(params: dict) -> dict:
    """
    Set full dict for deconvolution parameters independent of the peak bunch.

    Parameters
    ----------
    params : dict
        Dictionary containing some, not necessary all deconvolution parameters.

    Returns
    -------
    params : dict
        Dictionary containing all bunch-independent deconvolution parameters.

    """
    # TODO: Take care of "min_score", there should be way to estimate it
    # Select minimum accepted score between theoretical and experimental spectra
    min_score = params.pop("min_score", 150.0)

    # Select error tolerance between theoretical and experimental m/z;
    # perhaps this should be < 1./max(charge) for a reasonable value
    mass_error_tol = params.pop("mass_error_tol", 0.02)

    # Calculate goodness-of-fit criterion for envelope scoring
    params.setdefault("scorer", ms_ditp.MSDeconVFitter(min_score, mass_error_tol))

    # Set average composition of considered bases
    params.setdefault(
        "averagine", ms_ditp.Averagine(set_averagine(backbone="phosphate"))
    )

    # Set maximum number of tolerated missed peaks (keep small, i.e. 0 or 1)
    params.setdefault("max_missed_peaks", 0)

    # Set name of method to scale intensity value
    params.setdefault("scale_method", "sum")

    # Set error tolerance for matching experimental and theoretical peaks
    params.setdefault("error_tol", 2e-5)

    # TODO: Play with "truncate_after"; explore "incremental_truncation";
    #  perhaps also reduce the "min_score", such that multiple peaks
    #  with problems are coalesced into one
    # Set percentage of included isotopic pattern (very sensitive, see discussion in ms_deisotope docs)
    params.setdefault("truncate_after", 0.9)

    return params


def set_averagine(backbone: str) -> dict:
    """
    Calculate the average elemental composition of RNA.

    Parameters
    ----------
    backbone : ["no_backbone", "phosphate", "thiophosphate"]
        Backbone considered for the composition.

    Returns
    -------
    averagine_composition : dict
        Dictionary containing average elemental composition.

    """
    # Build dict with elemental compositions from file
    bases = pl.read_csv(
        importlib.resources.files(__package__) / "assets" / "elemental_composition.tsv",
        separator="\t",
    )
    base_compositions = [
        {
            col: row[bases.get_column_index(col)]
            for col in bases.columns
            if col != "base"
        }
        for row in bases.iter_rows()
    ]

    # Calculate average elemental composition
    average_composition = {}
    for element in base_compositions[0].keys():
        average_composition[element] = sum(
            [float(base[element]) for base in base_compositions]
        ) / len(base_compositions)

    # Add backbone elements (if needed)
    match backbone:
        case "no_backbone":
            pass
        case "phosphate":
            # Add 1 phosphorus and 2 oxygen for the phosphate group
            average_composition["O"] += 2
            average_composition["P"] += 1
        case "thiophosphate":
            # Add 1 phosphorus, 1 sulfur, and 1 oxygen for the phosphate group
            average_composition["O"] += 1
            average_composition["S"] += 1
            average_composition["P"] += 1
        case _:
            raise NotImplementedError(
                f"Support for '{backbone}' is currently not given."
            )

    return average_composition


def deconvolute_scans(file_path: str, params: dict) -> pl.DataFrame:
    """
    Deconvolute and deisotope MS2 scans from ThermoFisher RAW files and output
    a Polars dataframe containing deisotoped peaks.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.
    params : dict
        Dictionary containing deconvolution parameters.

    Returns
    -------
    polars.DataFrame
        Dataframe containing fragment monoisotopic masses and intensities.

    """
    # Load deconvolution parameter based on parameter dict
    params = DeconvolutionParameters(params)

    # Initialize the first scan from an iterator of the RAW file
    raw_file_read = initialize_raw_file_iterator(file_path=file_path)
    bunch = next(raw_file_read)

    peak_list = []
    for _ in tqdm.tqdm(range(len(raw_file_read) - 1), desc="Deisotoping MS2 scans"):
        # Only consider MS2 scans
        if bunch.ms_level == 2:
            # Centroid the scans
            bunch.pick_peaks()

            # Deisotope the scan and generate the dataframe of monoisotopic masses
            peak_list += deconvolute_scan(bunch=bunch, params=params)

        # Move on to the next scan
        bunch = next(raw_file_read)

    return aggregate_peaks_into_fragments(peak_list)


@dataclass
class DeisotopedPeak:
    scan_id: int
    scan_time: float
    peak_idx: int
    intensity: float
    neutral_mass: float
    is_precursor: bool
    is_deisotoped: bool
    mz: float


def deconvolute_scan(
    bunch: ms_ditp.data_source.Scan, params: dict
) -> List[DeisotopedPeak]:
    """
    Deconvolute peaks from MS2 scan.

    Parameters
    ----------
    bunch : ms_deisotope.data_source.Scan
        ThermoFisher scan.
    params : dict
        Dictionary containing deconvolution parameters.

    Returns
    -------
    peak_list : List[DeisotopedPeak]
        List containing deconvoluted peak data.

    """
    # Deconvolute/deisotope with ms_deisotope
    peak_set = ms_ditp.deconvolute_peaks(
        bunch,
        charge_range=params.charge_range
        if params.charge_range is not None
        else select_charge_range(bunch),
        minimum_intensity=params.minimum_intensity
        if params.minimum_intensity is not None
        else select_min_intensity(bunch),
        **params.bunch_independent_params,
    ).peak_set

    # Return None if scan does not contain any deisotoped peaks
    if len(peak_set) <= 0:
        return []

    # Obtain scan time (in min) and scan ID
    scan_time = bunch.scan_time * 60
    scan_id = int(bunch.scan_id.split("scan=")[-1])

    # Calculate m/z of precursor and accepted m/z range
    precursor_mz = bunch.precursor_information.mz
    min_mz = bunch.isolation_window.target - (1 * bunch.isolation_window.lower)
    max_mz = bunch.isolation_window.target + (1 * bunch.isolation_window.upper)

    # Iterate through the deisotoped scan
    peak_list = [0] * len(peak_set)
    for idx in range(len(peak_set)):
        mz = peak_set.peaks[idx].mz
        is_precursor = min_mz <= mz <= max_mz
        peak_list[idx] = DeisotopedPeak(
            scan_id=scan_id,
            scan_time=scan_time,
            peak_idx=idx,
            intensity=peak_set.peaks[idx].intensity,
            neutral_mass=peak_set.peaks[idx].neutral_mass,
            is_precursor=is_precursor,
            is_deisotoped=(
                False
                if not is_precursor
                else precursor_mz
                - abs(
                    ISOTOPIC_SHIFT_FACTOR
                    * ms_ditp.averagine.isotopic_shift(peak_set.peaks[idx].charge)
                )
                <= mz
                <= precursor_mz
            ),
            mz=mz,
        )
    return peak_list


def select_charge_range(bunch: ms_ditp.data_source.Scan) -> Tuple[int, int]:
    """
    Select range for accepted charge values.

    Parameters
    ----------
    bunch : ms_deisotope.data_source.Scan
        ThermoFisher scan.

    Returns
    -------
    min_charge : int
        Minimum accepted charge value.
    max_charge : int
        Maximum accepted charge value.

    """
    # Select charge (or use default if not given)
    charge = bunch.precursor_information.charge
    if not isinstance(charge, int):
        charge = DEFAULT_CHARGE_VALUE

    # Return charge with consideration to polarity
    return bunch.polarity, charge * bunch.polarity


def select_min_intensity(bunch: ms_ditp.data_source.Scan) -> float:
    """
    Select minimum intensity value below which peaks are ignored.

    Parameters
    ----------
    bunch : ms_deisotope.data_source.Scan
        ThermoFisher scan.

    Returns
    -------
    float
        Minimum intensity value.

    """
    # Return minimum intensity found in peak set of scan
    return min(p.intensity for p in bunch.peak_set)
    # TODO: Let the user define a threshold and take maximum of it and the above


def aggregate_peaks_into_fragments(peak_list: List[DeisotopedPeak]) -> pl.DataFrame:
    """
    Aggregate deisotoped peaks into fragments by grouping based on similar mass.

    Build dataframe of deisotoped peaks, group the peaks by their mass
    (within PPM tolerance), and aggregate them by selecting the maximum
    observed mass and total observed intensity in each group as a fragment.

    Parameters
    ----------
    peak_list : List[DeisotopedPeak]
        List containing deconvoluted peak data.

    Returns
    -------
    peak_df : polars.DataFrame
        Dataframe containing fragment monoisotopic masses and intensities.

    """
    # Build dataframe from peak list
    peak_df = pl.DataFrame(
        data=np.array(
            [
                [peak.__dict__[key] for key in COL_TYPES_DEISOTOPED.keys()]
                for peak in peak_list
            ]
        ),
        schema=COL_TYPES_DEISOTOPED,
    )

    # Cluster peaks together when mass is within PPM tolerance of each other
    peak_df = peak_df.sort("neutral_mass").with_columns(
        (
            abs(pl.col("neutral_mass").shift(1) - pl.col("neutral_mass"))
            / pl.col("neutral_mass").shift(1)
        )
        .fill_null(0)
        .fill_nan(0)
        .gt(PPM_TOLERANCE / 1e6)
        .cum_sum()
        .alias("ppm_group")
    )

    # Aggregate by PPM group (assign maximum neutral mass and total intensity to each group)
    return (
        peak_df.group_by("ppm_group")
        .agg(
            neutral_mass=pl.col("neutral_mass").max(),
            intensity=pl.col("intensity").sum(),
            is_precursor_deisotoped=pl.col("is_deisotoped").max(),
        )
        .sort("neutral_mass")
    )
