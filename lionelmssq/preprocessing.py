import ms_deisotope as ms_ditp
import yaml
from clr_loader import get_mono

from lionelmssq.deconvolution import deconvolute_scans
from lionelmssq.singleton_matching import match_singletons

rt = get_mono()

PPM_TOLERANCE = 10


def oliglow_run(
    filepath, deconv_params, meta_params, save_files=True, identify_singletons=True
):
    """
    Main pipeline for deconvoluting MS2 scans and generating the metafile required for running lionelmssq.
    Generates list of candidate nucleotides from singletons.
    Outputs a .tsv of the monoisotopic masses and a .YAML of the metadata in the working file directory.
    """
    raw_file_read = ms_ditp.data_source.thermo_raw_net.ThermoRawLoader(
        filepath, _load_metadata=True
    )

    df_deconvolved_agg, df_mz, sequence_mass = deconvolute_scans(
        raw_file_read, deconv_params, extract_mz=True
    )

    if identify_singletons:
        df_singletons = match_singletons(df_mz)
    else:
        df_singletons = None

    # Assumes that the file path has the format `path/to/file/sample_name.RAW`
    sample_name = filepath.split("/")[-1].split(".")[0]

    meta = create_metafile(
        sample_name,
        meta_params["intensity_cutoff"],
        meta_params["label_mass_3T"],
        meta_params["label_mass_5T"],
        meta_params.get("sequence_mass", sequence_mass),
        meta_params.get("true_sequence", None),
    )

    if save_files:
        with open(f"{sample_name}.meta.yaml", "w") as file:
            yaml.dump(meta, file)

        df_deconvolved_agg.write_csv(f"{sample_name}.tsv", separator="\t")
        if identify_singletons:
            df_singletons.write_csv(f"{sample_name}_singletons.tsv", separator="\t")
    else:
        return df_deconvolved_agg, df_singletons, meta


def create_metafile(
    sample_name,
    intensity_cutoff,
    label_mass_3T,
    label_mass_5T,
    sequence_mass,
    true_sequence=None,
):
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
