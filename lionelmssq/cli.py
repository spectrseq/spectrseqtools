from pathlib import Path
from typing import Literal

from lionelmssq.fragment_classification import classify_fragments
from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.masses import (
    COMPRESSION_RATE,
    MATCHING_THRESHOLD,
    TOLERANCE,
    build_breakage_dict,
    initialize_nucleotide_df,
)
from lionelmssq.prediction import Predictor
from tap import Tap
import polars as pl
import yaml


class Settings(Tap):
    fragments: Path  # path to .tsv table with observed fragments to use for prediction
    fragment_predictions: (
        Path  # path to .tsv table that shall contain the per fragment predictions
    )
    sequence_prediction: (
        Path  # path to .fasta file that shall contain the predicted sequence
    )
    sequence_name: str
    modification_rate: float = 0.5  # maximum percentage of modification in sequence
    solver: Literal["gurobi", "cbc"] = (
        "gurobi"  # solver to use for the optimization problem
    )
    threads: int = 1  # number of threads to use for the optimization problem


def main():
    settings = Settings(underscores_to_dashes=True).parse_args()

    solver_params = {
        "solver": select_solver(settings.solver),
        "threads": settings.threads,
        "msg": False,
    }

    # Read fragments
    fragments = pl.read_csv(settings.fragments, separator="\t")

    # Read additional parameter from meta file
    fragment_dir = settings.fragments.parent
    file_prefix = settings.fragments.stem
    with open(fragment_dir / f"{file_prefix}.meta.yaml", "r") as f:
        meta = yaml.safe_load(f)

    intensity_cutoff = meta["intensity_cutoff"] if "intensity_cutoff" in meta else 1e4
    start_tag = meta["label_mass_5T"] if "label_mass_5T" in meta else 555.1294
    end_tag = meta["label_mass_3T"] if "label_mass_3T" in meta else 455.1491

    simulation = False
    reduce_table = False
    reduce_set = True
    if "observed_mass" in fragments.columns:
        simulation = True
        reduce_table = True
        reduce_set = False

    _, unique_masses, explanation_masses = initialize_nucleotide_df(
        reduce_set=reduce_set
    )
    threshold = MATCHING_THRESHOLD if simulation else max(MATCHING_THRESHOLD, 20e-6)

    # Build breakage dict
    breakage_dict = build_breakage_dict(
        mass_5_prime=start_tag,
        mass_3_prime=end_tag,
    )

    # Standardize sequence mass (remove START_END breakage to gain SU mass)
    seq_mass_obs = meta["sequence_mass"]
    seq_mass_su = (
        seq_mass_obs
        - [
            mass * TOLERANCE
            for mass in breakage_dict
            if "START_END" in breakage_dict[mass]
        ][0]
    )

    dp_table = DynamicProgrammingTable(
        nucleotide_df=explanation_masses,
        compression_rate=int(COMPRESSION_RATE),
        tolerance=threshold,
        precision=TOLERANCE,
        modification_rate=settings.modification_rate,
        seq_mass_su=seq_mass_su,
        seq_mass_obs=seq_mass_obs,
        reduced_table=reduce_table,
        reduced_set=reduce_set,
    )

    fragments = classify_fragments(
        fragment_masses=fragments,
        dp_table=dp_table,
        breakage_dict=breakage_dict,
        output_file_path=fragment_dir / f"{file_prefix}.standard_unit_fragments.tsv",
        intensity_cutoff=intensity_cutoff,
    )

    prediction = Predictor(
        dp_table=dp_table,
        explanation_masses=explanation_masses,
    ).predict(
        fragments=fragments,
        solver_params=solver_params,
    )

    # save fragment predictions
    prediction.fragments.write_csv(settings.fragment_predictions, separator="\t")

    # save predicted sequence
    with open(settings.sequence_prediction, "w") as f:
        print(f">{settings.sequence_name}", file=f)
        print("".join(prediction.sequence), file=f)


def select_solver(solver: str):
    match solver:
        case "gurobi":
            return "GUROBI_CMD"
        case "cbc":
            return "PULP_CBC_CMD"
        case _:
            raise NotImplementedError(f"Support for '{solver}' is currently not given.")
