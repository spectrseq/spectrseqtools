import polars as pl
from typing import Tuple

from lionelmssq.deconvolution import deconvolute_scans
from lionelmssq.singleton_matching import match_singletons


def preprocess(
    file_path: str,
    deconvolution_params: dict,
    meta_params: dict,
    identify_singletons: bool = True,
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
    identify_singletons : bool, optional
        Flag whether to identify singletons from data. Default: True

    Returns
    -------
    df_deconvoluted : pl.DataFrame
        Dataframe containing deconvoluted fragments.
    df_singletons : pl.DataFrame
        Dataframe containing singleton data.
    meta : dict
        Dictionary with updated meta parameters.

    """
    # Deconvolute raw data from file
    df_deconvoluted, df_mz = deconvolute_scans(
        file_path=str(file_path),
        params=deconvolution_params,
        extract_mz=True,
    )

    # Identify singletons if desired
    df_singletons = match_singletons(df_mz=df_mz) if identify_singletons else None

    # Update meta parameters (if needed)
    meta_params.setdefault("identity", file_path.stem)
    meta_params.setdefault("sequence_mass", select_sequence_mass(df_deconvoluted))
    meta_params.setdefault("true_sequence", None)

    return df_deconvoluted, df_singletons, meta_params


def select_sequence_mass(df_deconvoluted: pl.DataFrame) -> float:
    """
    Select sequence mass from deconvoluted fragments.

    Determine the aggregated neutral_mass with (1) a deisotoped precursor and
    (2) the largest aggregated intensity as estimated intact sequence mass.

    Parameters
    ----------
    df_deconvoluted : pl.DataFrame
        Dataframe containing deconvoluted fragments.

    Returns
    -------
    float
        Sequence mass estimation.

    """
    return (
        df_deconvoluted.filter(pl.col("is_precursor_deisotoped"))
        .filter(pl.col("intensity") == pl.col("intensity").max())["neutral_mass"]
        .to_list()[0]
    )
