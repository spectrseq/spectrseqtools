import importlib.resources
import ms_deisotope as ms_ditp
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono
from typing import Tuple

from lionelmssq.singleton_matching import generate_mz_df

rt = get_mono()

PPM_TOLERANCE = 10


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


def deconvolute_scans(
    file_path: str, params: dict, extract_mz: bool = True
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Deconvolute and deisotope MS2 scans from ThermoFisher RAW files and output
    a Polars dataframe containing deisotoped peaks.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.
    params : dict
        Dictionary containing deconvolution parameters.
    extract_mz : bool, optional
        If true, extract m/z values per scan for singleton identification. The default is True.

    Returns
    -------
    df_deconvoluted : pl.DataFrame
        Dataframe containing monoisotopic masses and intensities.
    df_mz : pl.DataFrame
        Dataframe containing mass-to-charge ratios and intensities.

    """
    # Load deconvolution parameter based on parameter dict
    params = DeconvolutionParameters(params)

    # Read data from raw file
    raw_file_read = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(
        file_path, _load_metadata=True
    )

    # Initialize the raw_file_read iterator while ungrouping the MS1 from the MS2 scans
    raw_file_read.make_iterator(grouped=False)

    # Initialize the first scan from the iterator
    bunch = next(raw_file_read)

    decon_list = []
    mz_list = []
    for _ in tqdm.tqdm(range(len(raw_file_read) - 1), desc="Deisotoping MS2 scans"):
        # Only consider MS2 scans
        if bunch.ms_level == 2:
            # Centroid the scans
            bunch.pick_peaks()

            # Deisotope the scan and generate the dataframe of monoisotopic masses
            decon_df = generate_decon_df(bunch=bunch, params=params)

            if decon_df is not None:
                decon_list.append(decon_df)

            # Extract and generate dataframe of m/z for each scan (without deisotoping)
            if extract_mz:
                mz_df = generate_mz_df(bunch)
                if mz_df is not None:
                    mz_list.append(mz_df)

        # Move on to the next scan
        bunch = next(raw_file_read)

    # Build the final dataframe
    df_deconvoluted = pl.concat(decon_list)

    # Intact sequence mass calculation
    # Generate cluster indices when mass is within PPM_TOLERANCE of each other
    df_deconvoluted = df_deconvoluted.sort("neutral_mass").with_columns(
        (
            1e6
            * abs(pl.col("neutral_mass").shift(1) - pl.col("neutral_mass"))
            / pl.col("neutral_mass").shift(1)
        )
        .fill_null(0)
        .fill_nan(0)
        .gt(PPM_TOLERANCE)
        .cum_sum()
        .alias("ppm_group")
    )

    # Aggregate the final dataframe by PPM group while taking the maximum
    # neutral_mass and sum of the intensities for each group
    df_deconvoluted_agg = (
        df_deconvoluted.group_by("ppm_group")
        .agg(
            neutral_mass=pl.col("neutral_mass").max(),
            intensity=pl.col("intensity").sum(),
            is_precursor_deisotoped=pl.col("is_precursor_deisotoped").max(),
        )
        .sort("neutral_mass")
    )

    # Post-processing for df_mz
    if extract_mz:
        # Generate cluster indices when m/z is within PPM_TOLERANCE of each other
        df_mz = pl.concat(mz_list)
        df_mz = df_mz.sort("mz").with_columns(
            (1e6 * abs(pl.col("mz").shift(1) - pl.col("mz")) / pl.col("mz").shift(1))
            .fill_null(0)
            .fill_nan(0)
            .gt(PPM_TOLERANCE)
            .cum_sum()
            .alias("ppm_group")
        )
    else:
        df_mz = None

    return df_deconvoluted_agg, df_mz


def generate_decon_df(bunch, params):
    # Main deconvolution/deisotoping function from ms_deisotope
    t = ms_ditp.deconvolute_peaks(
        bunch,
        charge_range=params.charge_range
        if params.charge_range is not None
        else select_charge_range(bunch),
        minimum_intensity=params.minimum_intensity
        if params.minimum_intensity is not None
        else select_min_intensity(bunch),
        **params.bunch_independent_params,
    ).peak_set

    # Obtain scan ID and time
    scan_id = int(bunch.scan_id.split("scan=")[-1])
    scan_time = bunch.scan_time * 60

    # Calculate m/z isolation window
    iso = bunch.isolation_window
    min_mz = iso.target - (1 * iso.lower)
    max_mz = iso.target + (1 * iso.upper)

    precursor_mz = bunch.precursor_information.mz

    if len(t) > 0:
        # Initialize lists containing deisotoped information
        t_peak_index = [0] * len(t)
        t_intensity = [0] * len(t)
        t_neutral_mass = [0] * len(t)
        t_is_precursor = [False] * len(t)
        t_is_deisotoped = [False] * len(t)
        t_mz = [0] * len(t)

        # Iterate through the deisotoped scan
        for i in range(len(t)):
            t_peak_index[i] = i
            t_intensity[i] = t.peaks[i].intensity
            t_neutral_mass[i] = t.peaks[i].neutral_mass
            t_mz[i] = t.peaks[i].mz

            # Peak refers to a precursor scan if the m/z is within the isolation window
            if min_mz < t_mz[i] < max_mz:
                t_is_precursor[i] = True

                # Attempt to quality check the deisotoping step by checking if
                # the precursor isotope envelope has been deisotoped
                # `is_deisotoped` only refers to the precursor isotope envelope,
                # since this is the only envelope that we can reliably obtain
                # (since we have a reference m/z)
                # A precursor isotope envelope is deisotoped if the peak is:
                # (1) less than the precursor m/z, and
                # (2) greater than 10 times the isotopic shift
                shift = abs(ms_ditp.averagine.isotopic_shift(t.peaks[i].charge))
                if precursor_mz - (10 * shift) < t_mz[i] <= precursor_mz:
                    t_is_deisotoped[i] = True

        # Build the deisotoped dataframe for a scan
        decon_df = pl.DataFrame(
            data={
                "scan_id": scan_id,
                "scan_time": scan_time,
                "peak_index": t_peak_index,
                "intensity": t_intensity,
                "neutral_mass": t_neutral_mass,
                "is_precursor": t_is_precursor,
                "is_precursor_deisotoped": t_is_deisotoped,
                "mz": t_mz,
            }
        )

        return decon_df
    else:
        return None


def select_charge_range(bunch):
    # The maximum considered charge cannot be greater than that of the MS1 precursor charge!
    # Be careful of the polarity of the charge!
    # If the charges are negative, the charge range needs to be supplied with negative sign!
    charge = bunch.precursor_information.charge
    if isinstance(charge, int):
        return bunch.polarity, charge * bunch.polarity
    else:
        # TODO: Estimate this from the sequence length
        # IMP: This number should be large enough to cover the charge states of the precursors!
        return (
            bunch.polarity,
            bunch.polarity * 30,
        )


def select_min_intensity(bunch):
    # minimum_intensity = 5. #Default = 5, ignore peaks below this intensity!
    # Modify this based on the spectra!
    return min(p.intensity for p in bunch.peak_set)
    # Also, let the user define a threshold for this! and take the maximum of the above and this value!
