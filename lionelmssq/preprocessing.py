import polars as pl
from typing import Tuple

from lionelmssq.deconvolution import deconvolute_scans
from lionelmssq.singleton_matching import match_singletons

PPM_TOLERANCE = 10


def oliglow_run(
    file_path: str,
    deconvolution_params: dict,
    meta_params: dict,
    identify_singletons: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    Deconvolute MS2 scans and identify singletons.

    Main pipeline for deconvoluting MS2 scans and generating the metafile
    required for running LionelMSSQ.
    Generates list of candidate nucleotides from singletons.
    Outputs a TSV of the monoisotopic masses and a YAML of the metadata in
    the working file directory.

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.
    deconvolution_params : dict
        Dictionary with parameters for deconvolution.
    meta_params : dict
            Dictionary with meta parameters.
    identify_singletons : bool
        Flag whether to identify singletons from data.

    Returns
    -------
    df_deconvoluted_agg : pl.DataFrame
        Dataframe containing deconvoluted fragments.
    df_singletons : pl.DataFrame
        Dataframe containing singleton data.
    meta : dict
        Dictionary with updated meta parameters.

    """
    # Deconvolute raw data from file
    df_deconvoluted_agg, df_mz, sequence_mass = deconvolute_scans(
        file_path, deconvolution_params, extract_mz=True
    )
    print(df_deconvoluted_agg)

    if identify_singletons:
        df_singletons = match_singletons(df_mz)
    else:
        df_singletons = None

    # Assumes that the file path has the format `path/to/file/sample_name.RAW`
    sample_name = file_path.split("/")[-1].split(".")[0]

    meta = create_metafile(
        sample_name=sample_name,
        intensity_cutoff=meta_params["intensity_cutoff"],
        start_tag=meta_params["label_mass_5T"],
        end_tag=meta_params["label_mass_3T"],
        sequence_mass=meta_params.get("sequence_mass", sequence_mass),
        true_sequence=meta_params.get("true_sequence", None),
    )

    return df_deconvoluted_agg, df_singletons, meta


def create_metafile(
    sample_name: str,
    intensity_cutoff: float,
    start_tag: float,
    end_tag: float,
    sequence_mass: float,
    true_sequence: bool = None,
) -> dict:
    """
    Create the metafile required for running lionelmssq. Contains experimental information from the sample.

    Parameters
    ----------
    sample_name : str
        Sample ID.
    intensity_cutoff : float
        Minimum intensity considered in LionelMSSQ.
    start_tag : float
        Mass of the 5'-label of the sample.
    end_tag : float
        Mass of the 3'-label of the sample.
    sequence_mass : float
        Estimated intact sequence mass.
    true_sequence : str, optional
        True sequence of the sample, if available.

    Returns
    -------
    meta : dict
        Dictionary containing metadata, to be saved as a YAML file.

    """

    meta = {
        "identity": str(sample_name),
        "intensity_cutoff": float(intensity_cutoff),
        "label_mass_3T": float(end_tag),
        "label_mass_5T": float(start_tag),
        "sequence_mass": float(sequence_mass),
        "true_sequence": str(true_sequence),
    }

    return meta
