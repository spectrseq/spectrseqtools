import numpy as np
import polars as pl
import tqdm
import yaml
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs

from spectrseqtools.dataclasses import Sequence
from spectrseqtools.file_settings import load_alphabet
from spectrseqtools.parsers import MixturePostprocessingOptions


def evaluate_mixture(options: MixturePostprocessingOptions) -> None:
    with open(options.meta, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
        if "true_sequence" not in meta:
            meta["true_sequence"] = "".join(meta["true_sequences"])

    # Load predictions fasta file and generate sequence dictionary, mapping MS1 group
    # number to prediction
    sequence_dict = {}
    with open(str(options.prediction), mode="r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in zip(lines[::2], lines[1::2]):
            head, seq = line

            assert head.startswith(">")
            if head.endswith("_full\n"):
                continue

            grp_number = int(head.split(".")[-1].removesuffix("\n"))
            sequence = Sequence.from_str(seq)
            if len(sequence.sequence) == 0:
                continue
            sequence_dict[grp_number] = sequence

    # Get list of intact masses
    fragments = pl.read_csv(options.fragments, separator="\t")
    ms1_masses = fragments.filter(
        pl.col("is_ms1_mass")
        & pl.col("ms1_mass_group").is_in(list(sequence_dict.keys()))
    ).with_columns(
        pl.col("min_window_time").cast(pl.Float64),
        pl.col("max_window_time").cast(pl.Float64),
    )

    masses = load_alphabet()

    # Generate prediction list for alignment
    prediction_vals = []

    for grp_number, seq in sequence_dict.items():
        ms1_mass_info = ms1_masses.filter(pl.col("ms1_mass_group") == grp_number)
        assert len(ms1_mass_info) == 1
        ms1_mass_info = ms1_mass_info[0]

        prediction_vals.append(
            {
                "group_number": grp_number,
                "intact_mass": ms1_mass_info["observed_mass"][0],
                "predicted_sequence": seq.to_encoding(masses),
                "min_window_time": ms1_mass_info["min_window_time"][0],
                "max_window_time": ms1_mass_info["max_window_time"][0],
                "adduct_types": ms1_mass_info["adduct_type"][0],
            }
        )

    def compare_prediction_to_target(
        prediction_raw, target_sequence, is_backward=False
    ):
        if is_backward:
            prediction_raw = prediction_raw[::-1]
        prediction_len = len(prediction_raw)
        prediction_string = "".join(prediction_raw)

        target_strings = []

        for i in range(len(target_sequence) - prediction_len + 1):
            target_sequence_window = target_sequence[i : i + prediction_len]
            target_strings.append(target_sequence_window)

        distances = normalized_damerau_levenshtein_distance_seqs(
            prediction_string, target_strings
        )
        min_distance_idx = np.argmin(distances)

        min_distance = distances[min_distance_idx]
        start_pos = min_distance_idx
        end_pos = min_distance_idx + prediction_len

        return (
            prediction_string,
            target_strings[min_distance_idx],
            min_distance,
            start_pos,
            end_pos,
            is_backward,
        )

    def align_prediction_results(
        prediction_vals, target_sequence, alignment_score_threshold=0
    ):

        alignment_vals = []

        for i in tqdm.tqdm(prediction_vals, desc="Calculating alignment scores"):
            prediction_raw = i["predicted_sequence"]
            if len(prediction_raw) == 0:
                continue

            forward_comparison = compare_prediction_to_target(
                prediction_raw, target_sequence, is_backward=False
            )
            backward_comparison = compare_prediction_to_target(
                prediction_raw, target_sequence, is_backward=True
            )

            final_comparison = (
                forward_comparison
                if forward_comparison[2] <= backward_comparison[2]
                else backward_comparison
            )

            if final_comparison[2] > alignment_score_threshold:
                continue

            alignment_vals.append(
                final_comparison
                + (
                    i["group_number"],
                    i["intact_mass"],
                    i["min_window_time"],
                    i["max_window_time"],
                    i["adduct_types"],
                )
            )

        df_alignment = pl.DataFrame(
            alignment_vals,
            schema=[
                "predicted_string",
                "best_matching_target_string",
                "normalized_damerau_levenshtein_distance",
                "target_start_pos",
                "target_end_pos",
                "is_backward",
                "group",
                "intact_mass",
                "min_window_time",
                "max_window_time",
                "adduct_type",
            ],
            orient="row",
        )

        return df_alignment

    def interpret_alignment_results(df_alignment, target_sequence):

        match_rows = []

        for r in df_alignment.iter_rows(named=True):
            pred_string = r["predicted_string"]
            target_string = r["best_matching_target_string"]
            start_pos = r["target_start_pos"]
            intact_mass = r["intact_mass"]
            min_window_time = r["min_window_time"]
            max_window_time = r["max_window_time"]
            adduct_type = r["adduct_type"]

            for i, (p, t) in enumerate(zip(pred_string, target_string)):
                if p == t:
                    status = "match"
                else:
                    status = "mismatch"
                match_rows.append(
                    {
                        "group": r["group"],
                        "score": r["normalized_damerau_levenshtein_distance"],
                        "target_position": start_pos + i,
                        "predicted_base": p,
                        "target_base": t,
                        "status": status,
                        "intact_mass": intact_mass,
                        "min_window_time": min_window_time,
                        "max_window_time": max_window_time,
                        "adduct_type": adduct_type,
                    }
                )

        for idx, entry in enumerate(target_sequence):
            match_rows.append(
                {
                    "group": -1,
                    "score": 0,
                    "target_position": idx,
                    "predicted_base": entry[0],
                    "target_base": entry[0],
                    "status": "match",
                    "intact_mass": np.nan,
                    "min_window_time": np.nan,
                    "max_window_time": np.nan,
                    "adduct_type": "",
                }
            )

        df_expanded_alignment = pl.DataFrame(match_rows)

        df_expanded_alignment = df_expanded_alignment.sort(["group", "target_position"])

        df_expanded_alignment = df_expanded_alignment.with_columns(
            [
                pl.col("status").shift(-1).over("group").alias("next_status"),
                pl.col("predicted_base").shift(-1).over("group").alias("next_pred"),
                pl.col("target_base").shift(-1).over("group").alias("next_target"),
            ]
        )

        df_expanded_alignment = df_expanded_alignment.with_columns(
            (
                (pl.col("status") == "mismatch")
                & (pl.col("next_status") == "mismatch")
                & (pl.col("predicted_base") == pl.col("next_target"))
                & (pl.col("next_pred") == pl.col("target_base"))
            ).alias("is_swap_start")
        )

        df_expanded_alignment = df_expanded_alignment.with_columns(
            (
                pl.col("is_swap_start") | pl.col("is_swap_start").shift(1).over("group")
            ).alias("is_swap")
        )

        df_expanded_alignment = df_expanded_alignment.with_columns(
            pl.when(pl.col("is_swap"))
            .then(pl.lit("swap"))
            .otherwise(pl.col("status"))
            .alias("status")
        )

        df_expanded_alignment = df_expanded_alignment.drop(
            ["next_status", "next_pred", "next_target", "is_swap_start", "is_swap"]
        )

        df_expanded_alignment = df_expanded_alignment.join(
            masses.select(["encoding", "canonical_name"]),
            left_on="predicted_base",
            right_on="encoding",
            how="left",
        )

        df_expanded_alignment = df_expanded_alignment.sort(
            ["score", "group", "target_position"]
        )
        group_to_pos = {
            group: i
            for i, group in enumerate(
                df_expanded_alignment["group"].unique(maintain_order=True).to_list()
            )
        }
        df_expanded_alignment = df_expanded_alignment.with_columns(
            pos=pl.col("group").replace(group_to_pos)
        )

        return df_expanded_alignment

    # Standardized reference sequence to be used in alignment at the last step
    target_sequence = "".join(
        Sequence.from_str(meta["true_sequence"]).to_encoding(masses)
    )

    # Align predicted sequences to reference sequence
    df_alignment = align_prediction_results(prediction_vals, target_sequence, 1)
    df_expanded_alignment = interpret_alignment_results(df_alignment, target_sequence)

    df_alignment.write_csv(options.output_path / "df_alignment.csv", separator=",")
    df_expanded_alignment.write_csv(
        options.output_path / "df_expanded_alignment.csv", separator=","
    )
    return df_expanded_alignment
