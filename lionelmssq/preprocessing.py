import polars as pl
from typing import Tuple

from lionelmssq.deconvolution import deconvolute
from lionelmssq.singleton_identification import identify_singletons


def preprocess(
    file_path: str,
    deconvolution_params: dict,
    meta_params: dict,
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    Deconvolute MS2 scans and identify singletons.

    Main pipeline for deconvoluting MS2 scans and generating the metafile
    required for running LionelMSSQ as well as a list of candidate nucleotides
    from singletons (if desired).

    Parameters
    ----------
    file_path : str
        Path of RAW file from ThermoFisher.
    deconvolution_params : dict
        Dictionary with parameters for deconvolution.
    meta_params : dict
        Dictionary with meta parameters.

    Returns
    -------
    fragments : polars.DataFrame
        Dataframe containing deconvoluted fragments.
    singletons : polars.DataFrame
        Dataframe containing singleton data.
    meta_params : dict
        Dictionary with updated meta parameters.

    """
    # Deconvolute raw data from file
    fragments = deconvolute(
        file_path=str(file_path),
        params=deconvolution_params,
    )

    # Identify singletons
    singletons = identify_singletons(file_path=str(file_path))

    # Update meta parameters (if needed)
    meta_params.setdefault("identity", file_path.stem)
    meta_params.setdefault("sequence_mass", select_sequence_mass(fragments))
    meta_params.setdefault("true_sequence", None)

    return fragments, singletons, meta_params


def select_sequence_mass(fragments: pl.DataFrame) -> float:
    """
    Select sequence mass from deconvoluted fragments.

    Determine the aggregated neutral_mass with (1) a deisotoped precursor and
    (2) the largest aggregated intensity as estimated intact sequence mass.

    Parameters
    ----------
    fragments : polars.DataFrame
        Dataframe containing deconvoluted fragments.

    Returns
    -------
    float
        Sequence mass estimation.

    Notes
    -----
    This function is inspired by https://github.com/koesterlab/oliglow,
    originally implemented by Moshir Harsh (btemoshir@gmail.com).

    """
    return (
        fragments.filter(pl.col("is_precursor_deisotoped"))
        .filter(pl.col("intensity") == pl.col("intensity").max())["neutral_mass"]
        .to_list()[0]
    )
