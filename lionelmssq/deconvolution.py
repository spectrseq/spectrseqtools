import ms_deisotope as ms_ditp
import polars as pl
import tqdm as tqdm
from clr_loader import get_mono

from lionelmssq.singleton_matching import generate_mz_df

rt = get_mono()

PPM_TOLERANCE = 10


class DeconvolutionParameters:
    def __init__(self, parameters):
        self.averagine = parameters.get(
            "averagine", ms_ditp.Averagine(create_averagine())
        )
        # The minimum score between the theo and exp spectra using MSDeconVFitter which is accepted!
        # TODO: Take care of this score, there should be way to estimate this score!
        # Or use multiple passes, if the score is too large, i.e. the number
        # of peaks after deisotoping are less than \alpha (num peaks), increase this value!
        self.min_score = parameters.get("min_score", 150.0)
        # The Absolute error tolerance between the theoretical m/z and the exp m/z which is accepted!
        # perhaps this should be < 1./max(charge) for a reasonable value.
        self.mass_error_tol = parameters.get("mass_error_tol", 0.02)

        # The parameters are intensity scale dependent.
        # We will need to find this dependent on the intensity of the scans!

        # For MS1 and 0.8 for MS2 as per the recommendation of the ms_ditp author!
        # See the discussion in ms_isotope docs
        # SENSITIVE TO THIS PARAMETER! EXTREMELY SENSITIVE! Reduce this if high mass ions are missing!
        # The authors propose to use `incremental_truncation` if the latter is the case! Explore this later!
        # Play with the parameter Truncate_after! Perhaps also reduce the minimum score of the scorer, such that multiple peaks with problems are coalesced into one!
        self.truncate_after = parameters.get("truncate_after", 0.9)
        # What to do with the intensity of the different peaks!
        self.scale = parameters.get("scale", "sum")
        # If this parameter is increased then more deconvoluted peaks are selected, keep this small preferrentially (0 or 1)
        self.max_missed_peaks = parameters.get("max_missed_peaks", 0)
        # PPM error tolerance to use to match experimental to theoretical peaks, default = 2e-5
        self.error_tol = parameters.get("error_tol", 2e-5)
        # How to scale intensities when comparing theoretical to exp isotopic distributions, default = "sum"
        self.scale_method = parameters.get("scale_method", "sum")


def create_averagine(backbone: str):
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
    # Initialize element compositions for unmodified bases
    adenosine = {"C": 10, "H": 13, "N": 5, "O": 4}
    cytidine = {"C": 9, "H": 13, "N": 3, "O": 5}
    guanosine = {"C": 10, "H": 13, "N": 5, "O": 5}
    uridine = {"C": 9, "H": 12, "N": 2, "O": 6}

    # Determine average composition based on all considered bases
    base_compositions = [adenosine, cytidine, guanosine, uridine]
    elements = ["C", "H", "N", "O", "P", "S"]
    average_composition = {}
    for element in elements:
        for base in base_compositions:
            base.setdefault(element, 0)
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


def deconvolute_scans(file_path, parameters, extract_mz=True):
    """
    Deconvolute and deisotope MS2 scans from ThermoFisher RAW files and output
    a Polars dataframe containing deisotoped peaks and the estimated intact
    sequence mass.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.
    parameters : dict
        Dictionary of deisotoping parameters.
    extract_mz : bool, optional
        If true, extract m/z values per scan for singleton identification. The default is True.

    Returns
    -------
    df_deconvoluted : pl.DataFrame
        Dataframe containing monoisotopic masses and intensities.
    df_mz : pl.DataFrame
        Dataframe containing mass-to-charge ratios and intensities.
    sequence_mass : float
        Estimated intact sequence mass
    """
    # Read data from raw file
    raw_file_read = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(
        file_path, _load_metadata=True
    )

    # Load parameter defaults if not found in parameters dict
    params = DeconvolutionParameters(parameters)

    # Calculate goodness-of-fit criterion for envelope scoring
    scorer = ms_ditp.MSDeconVFitter(params.min_score, params.mass_error_tol)

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

            # Compute additional parameters if they are not included in the parameters dict
            charge_range = parameters.get("charge_range", default_charge_range(bunch))
            minimum_intensity = parameters.get(
                "minimum_intensity", default_min_intensity(bunch)
            )

            # Deisotope the scan and generate the dataframe of monoisotopic masses
            decon_df = generate_decon_df(
                params, charge_range, minimum_intensity, scorer, bunch
            )

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

    # The aggregated neutral_mass with
    # (1) a deisotoped precursor and
    # (2) has the largest aggregated intensity is the estimated intact sequence mass
    sequence_mass = (
        df_deconvoluted_agg.filter(pl.col("is_precursor_deisotoped"))
        .filter(pl.col("intensity") == pl.col("intensity").max())["neutral_mass"]
        .to_list()[0]
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

    return df_deconvoluted_agg, df_mz, sequence_mass


def default_charge_range(bunch):
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


def default_min_intensity(bunch):
    # minimum_intensity = 5. #Default = 5, ignore peaks below this intensity!
    # Modify this based on the spectra!
    return min(p.intensity for p in bunch.peak_set)
    # Also, let the user define a threshold for this! and take the maximum of the above and this value!


def generate_decon_df(params, charge_range, minimum_intensity, scorer, bunch):
    # Main deconvolution/deisotoping function from ms_deisotope
    t = ms_ditp.deconvolute_peaks(
        bunch,
        averagine=params.averagine,
        scorer=scorer,
        max_missed_peaks=params.max_missed_peaks,
        charge_range=charge_range,
        truncate_after=params.truncate_after,
        scale=params.scale,
        error_tolerance=params.error_tol,
        minimum_intensity=minimum_intensity,
        scale_method=params.scale_method,
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
