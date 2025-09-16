import ms_deisotope as ms_ditp
import polars as pl
import tqdm as tqdm
from lionelmssq.singleton_matching import generate_mz_df, match_singletons
import yaml
from clr_loader import get_mono
rt = get_mono()

PPM_TOLERANCE = 10

# Defines the averagine for RNA with and without backbone
def create_averageine(with_backbone=True, with_thiophosphate_backbone=False):
    """
    Calculate the average elemental composition of RNA nucleosides, optionally including the backbone.
    Args:
        with_backbone (bool): If True, include the phosphate group in the calculation. Defaults to True.
    Returns:
        dict: A dictionary representing the average elemental composition of RNA nucleosides.
              The keys are the elements ("C", "H", "N", "O", "P") and the values are the average counts of each element.
              If with_backbone is True, the composition includes the phosphate group.
    """

    if with_thiophosphate_backbone:
        # Nucleosides:
        Adenosine = {"C": 10, "H": 13, "N": 5, "O": 4, "P": 0, "S": 0}
        Cytidine = {"C": 9, "H": 13, "N": 3, "O": 5, "P": 0, "S": 0}
        Guanosine = {"C": 10, "H": 13, "N": 5, "O": 5, "P": 0, "S": 0}
        Uridine = {"C": 9, "H": 12, "N": 2, "O": 6, "P": 0, "S": 0}

        elements = ["C", "H", "N", "O", "P", "S"]

    else:
        # Nucleosides:
        Adenosine = {"C": 10, "H": 13, "N": 5, "O": 4, "P": 0}
        Cytidine = {"C": 9, "H": 13, "N": 3, "O": 5, "P": 0}
        Guanosine = {"C": 10, "H": 13, "N": 5, "O": 5, "P": 0}
        Uridine = {"C": 9, "H": 12, "N": 2, "O": 6, "P": 0}

        elements = ["C", "H", "N", "O", "P"]

    nucleosides_composition = [Adenosine, Cytidine, Guanosine, Uridine]

    average_nucleoside_rna = {key: 0.0 for key in elements}

    for e in elements:
        average_nucleoside_rna[e] = sum(
            [float(nucleosides[e]) for nucleosides in nucleosides_composition]
        ) / len(nucleosides_composition)
    # print("RNA nucleoside averagine = ", average_nucleoside_rna)

    # for e in elements:
    
    if with_backbone:
        averagine_rna_with_backbone = average_nucleoside_rna.copy()
        averagine_rna_with_backbone["O"] += 2  # Add 2 oxygens for the phosphate group
        averagine_rna_with_backbone["P"] += 1  # Add 1 phosphorus for the phosphate group
        return averagine_rna_with_backbone
    
    elif with_thiophosphate_backbone:
        averagine_rna_with_thiophosphate_backbone = average_nucleoside_rna.copy()
        averagine_rna_with_thiophosphate_backbone["O"] += (
            1  # Add 1 oxygen for the phosphate group
        )
        averagine_rna_with_thiophosphate_backbone["S"] += (
            1  # Add 1 sulphur for the thiophosphate group
        )
        averagine_rna_with_thiophosphate_backbone["P"] += (
            1  # Add 1 phosphorus for the phosphate group
        )
        return averagine_rna_with_thiophosphate_backbone
    
    else:
        return average_nucleoside_rna

class DeconvolutionParameters:
    def __init__(self, parameters):
        self.averagine = parameters.get("averagine", 
                                        ms_ditp.Averagine(create_averageine()))
        # The minimum score between the theo and exp spectra using MSDeconVFitter which is accepted!
        # TODO: Take care of this score, there should be way to estimate this score!#
        # Or use multiple passes, if the score is too large, i.e. the number of peaks after desiotoping are less than \alpha (num peaks), increase this value!
        self.min_score = parameters.get("min_score", 
                                        150.0)
        self.mass_error_tol = parameters.get("mass_error_tol", 
                                             0.02)
        # The Absolute error tolerance between the theoretical m/z and the exp m/z which is accepted!
        # perhaps this should be < 1./max(charge) for a reasonable value.

        # The parameters are intensity scale depenedent.
        # We will need to find this dependant on the intensity of the scans!

        self.truncate_after = parameters.get("truncate_after", 
                                             0.8) # For MS1 and 0.8 for MS2 as per the recommendation of the ms_ditp author!
        # See the discussion in ms_isotope docs
        # SENSITIVE TO THIS PARAMETER! EXTREMELY SENSETIVE! Reduce this if high mass ions are missing!
        # The authors propose to use `incremental_truncation` if the latter is the case! Explore this later!
        # Play with the parameter Truncate_after! Perhaps also reduce the minimum score of the scorer, such that multiple peaks with problems are coalesced into one!
        self.scale = parameters.get("scale", 
                                    "sum") # What to do with the intensity of the different peaks!

        # If this paramter is increased then more deconvoluted peaks are selected, keep this small preferrentially (0 or 1)
        self.max_missed_peaks = parameters.get("max_missed_peaks",
                                                0)

        self.error_tol = parameters.get("error_tol", 
                                        2e-5) # error_tol = 2e-5  # Also the default value! PPM error tolerance to use to match experimental to theoretical peaks, defalut = 2e-5

        self.scale_method = parameters.get("scale_method", 
                                           "sum") # scale_method = "max" #How to scale intensities when comparing theoretical to exp isotopic distributions, default = "sum"
        
def default_charge_range(bunch):
    # # The maximum considered charge cannot be greater than that of the MS1 precursor charge!
    # # Be careful of the polarity of the charge! If the charges are negative, the charge range needs to be supplied with negative sign!
    charge = bunch.precursor_information.charge
    if isinstance(charge, int):
        return (bunch.polarity, charge*bunch.polarity)
    else:
        return (bunch.polarity, bunch.polarity * 30) # TODO: Estimate this from the sequence length
            # IMP: This number should be large enough to cover the charge states of the precursors!

def default_min_intensity(bunch):
    # minimum_intensity = 5. #Defualt = 5, ignore peaks below this intensity! Modify this based on the speatra!
    return min(p.intensity for p in bunch.peak_set)
    # Also, let the user define a threshold for this! and take the maximum of the above and this value!

def generate_decon_df(params, charge_range, minimum_intensity, scorer, bunch):
    
    #Main deconvolution/deisotoping function from ms_deisotope
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
            scale_method=params.scale_method
            ).peak_set
    
    #Obtain scan ID and time
    scan_id = int(bunch.scan_id.split("scan=")[-1])
    scan_time = bunch.scan_time*60

    #Calculate m/z isolation window
    iso = bunch.isolation_window
    min_mz = iso.target - (1*iso.lower)
    max_mz = iso.target + (1*iso.upper)
    
    precursor_mz = bunch.precursor_information.mz

    if len(t)>0:
        
        #Initialize lists containing deisotoped information
        t_peak_index = [0]*len(t)
        t_intensity = [0]*len(t)
        t_neutral_mass = [0]*len(t)
        t_is_precursor = [False]*len(t)
        t_is_deisotoped = [False]*len(t)
        t_mz = [0]*len(t)
        
        #Iterate through the deisotoped scan
        for i in range(len(t)):
            t_peak_index[i] = i
            t_intensity[i] = t.peaks[i].intensity
            t_neutral_mass[i] = t.peaks[i].neutral_mass
            t_mz[i] = t.peaks[i].mz

            #Peak refers to a precursor scan if the m/z is within the isolation window
            if min_mz < t_mz[i] < max_mz:
                t_is_precursor[i] = True

                #Attempt to quality check the deisotoping step by checking if the precursor isotope envelope has been deisotoped
                #`is_deisotoped` only refers to the precursor isotope envelope, since this is the only envelope that we can reliably obtain (since we have a reference m/z)
                #A precursor isotope envelope is deisotoped if the peak is: (1) less than the precursor m/z, and (2) greater than 10 times the isotopic shift
                shift = abs(ms_ditp.averagine.isotopic_shift(t.peaks[i].charge))
                if precursor_mz-(10*shift) < t_mz[i] <= precursor_mz:
                    t_is_deisotoped[i] = True

        #Build the deisotoped dataframe for a scan
        decon_df = pl.DataFrame(data = {"scan_id": scan_id,
                                        "scan_time": scan_time,
                                        "peak_index": t_peak_index,
                                        "intensity": t_intensity,
                                        "neutral_mass": t_neutral_mass,
                                        "is_precursor": t_is_precursor,
                                        "is_precursor_deisotoped": t_is_deisotoped, #Only refers to the precursor isotope envelope
                                        "mz": t_mz})
    
        return decon_df
    else:
        return None

def deconvolute_scans(raw_file_read, parameters, extract_mz = True):
    """
    Deconvolute and deisotope MS2 scans from Thermo Fisher .RAW files and output a Polars dataframe containing deisotoped peaks and the estimated intact sequence mass.

    Parameters
    ----------
    raw_file_read : ms_deisotope.data_source.thermo_raw_net.ThermoRawLoader
        Thermo RAW iterator.
    parameters : dict
        Dictionary of deisotoping parameters.
    extract_mz : bool, optional
        If true, extract m/z values per scan for singleton identification. The default is True. 

    Returns
    -------
    df_deconvolved : polars dataframe
        Dataframe containing monoisotopic masses and intensities.
    df_mz : polars dataframe
        Dataframe containing mass-to-charge ratios and intensities.
    sequence_mass : float
        Estimated intact sequence mass
    """

    #Load parameter defaults if not found in `parameters` dict
    params = DeconvolutionParameters(parameters)
    #Calculate goodness-of-fit criterion for envelope scoring
    scorer = ms_ditp.MSDeconVFitter(params.min_score, 
                                    params.mass_error_tol)

    #Initialize the raw_file_read iterator while ungrouping the MS1 from the MS2 scans
    raw_file_read.make_iterator(grouped=False)
    #Initialize the first scan from the iterator
    bunch = next(raw_file_read)

    decon_list = []
    mz_list = []
    for _ in tqdm.tqdm(range(len(raw_file_read)-1), desc = "Deisotoping MS2 scans"):
        if bunch.ms_level == 2: #Only consider MS2 scans

            #Centroid the scans
            bunch.pick_peaks()
            #Compute additional parameters if they are not included in the `parameters` dict
            charge_range = parameters.get("charge_range", 
                                        default_charge_range(bunch))
            minimum_intensity = parameters.get("minimum_intensity", 
                                            default_min_intensity(bunch))

            #Deisotope the scan and generate the dataframe of monoisotopic masses
            decon_df = generate_decon_df(params,
                                         charge_range, 
                                         minimum_intensity, 
                                         scorer, 
                                         bunch)
            
            if decon_df is not None:
                decon_list.append(decon_df)

            #Extract and generate dataframe of m/z for each scan (without deisotoping)
            if extract_mz:
                mz_df = generate_mz_df(bunch)
                if mz_df is not None:
                    mz_list.append(mz_df)

        #Move on to the next scan
        bunch = next(raw_file_read)

    #Build the final dataframe
    df_deconvolved = pl.concat(decon_list)

    #Intact sequence mass calculation
    #Generate cluster indices when mass is within PPM_TOLERANCE of each other
    df_deconvolved = df_deconvolved.sort("neutral_mass").with_columns((1e6*abs(pl.col("neutral_mass").shift(1)
                                                                                -pl.col("neutral_mass"))
                                                                                /pl.col("neutral_mass").shift(1))
                                                                    .fill_null(0).fill_nan(0)
                                                                    .gt(PPM_TOLERANCE)
                                                                    .cum_sum().alias("ppm_group"))

    #Aggregate the final dataframe by `ppm_group` while taking the maximum `neutral_mass` and sum of the `intensities` for each group
    df_deconvolved_agg = df_deconvolved.group_by("ppm_group").agg(neutral_mass = pl.col("neutral_mass").max(), 
                                                                  intensity = pl.col("intensity").sum(), 
                                                                  is_precursor_deisotoped = pl.col("is_precursor_deisotoped").max())

    #The aggregated `neutral_mass` with (1) a deisotoped precursor and (2) has the largest aggregated `intensity` is the estimated intact sequence mass
    sequence_mass = (df_deconvolved_agg.filter(pl.col("is_precursor_deisotoped"))
                     .filter(pl.col("intensity") == pl.col("intensity").max())["neutral_mass"]
                     .to_list()[0])
    
    #Post-processing for df_mz
    #Generate cluster indices when m/z is within PPM_TOLERANCE of each other
    if extract_mz:
        df_mz = pl.concat(mz_list)
        df_mz = df_mz.sort("mz").with_columns((1e6*abs(pl.col("mz").shift(1)
                                                       -pl.col("mz"))
                                                       /pl.col("mz").shift(1))
                                            .fill_null(0).fill_nan(0)
                                            .gt(PPM_TOLERANCE)
                                            .cum_sum().alias("ppm_group"))
    else:
        df_mz = None

    return df_deconvolved_agg, df_mz, sequence_mass

def create_metafile(sample_name, 
                    intensity_cutoff, 
                    label_mass_3T, 
                    label_mass_5T, 
                    sequence_mass, 
                    true_sequence = None):
    """
    Create the metafile required for running lionelmssq. Contains experimental information from the sample.

    Parameters
    ----------
    sample_name : str
        Sample ID.
    intensity_cutoff : float
        Minimum intensity considered in lionelmssq.
    label_mass_3T : float
        Mass of the 3' label of the sample.
    label_mass_5T : float
        Mass of the 5' label of the sample.
    sqeuence_mass : float
        Estimated intact sequence mass, obtained from `deconvolute_scans`.
    true_sequence : str, optional
        True sequence of the sample, if available.
    file_prefix : str, optional
        Output filename of the metafile.

    Returns
    -------
    meta : dict
        Dictionary containing metadata, to be saved as a .YAML file.

    """

    meta = {
    "identity": str(sample_name),
    "intensity_cutoff": float(intensity_cutoff),
    "label_mass_3T": float(label_mass_3T),
    "label_mass_5T": float(label_mass_5T),
    "sequence_mass": float(sequence_mass),
    "true_sequence": str(true_sequence),
    }

    return meta

def oliglow_run(filepath, deconv_params, meta_params,
                save_files = True,
                identify_singletons = True):
    """
    Main pipeline for deconvoluting MS2 scans and generating the metafile required for running lionelmssq.
    Generates list of candidate nucleotides from singletons.
    Outputs a .tsv of the monoisotopic masses and a .YAML of the metadata in the working file directory.
    """
    raw_file_read = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(filepath, _load_metadata=True)

    df_deconvolved_agg, df_mz, sequence_mass = deconvolute_scans(raw_file_read, deconv_params, extract_mz = True)
    
    if identify_singletons:
        df_singletons = match_singletons(df_mz)
    else:
        df_singletons = None
    
    #Assumes that the file path has the format `path/to/file/sample_name.RAW`
    sample_name = filepath.split("/")[-1].split(".")[0]

    meta = create_metafile(sample_name,
                           meta_params["intensity_cutoff"],
                           meta_params["label_mass_3T"],
                           meta_params["label_mass_5T"],
                           meta_params.get("sequence_mass", sequence_mass),
                           meta_params.get("true_sequence", None))
    
    if save_files:
        with open(f"{sample_name}.meta.yaml", 'w') as file:
            yaml.dump(meta, file)

        df_deconvolved_agg.write_csv(f"{sample_name}.tsv", separator = "\t")
        if identify_singletons:
            df_singletons.write_csv(f"{sample_name}_singletons.tsv", separator = "\t")
    else:
        return df_deconvolved_agg, df_singletons, meta