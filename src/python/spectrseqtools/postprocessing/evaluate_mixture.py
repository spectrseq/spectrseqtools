# -*- coding: utf-8 -*-
"""Postprocessing of predictions for complex mixture by evaluation thereof."""

from typing import Tuple

import numpy as np
import polars as pl
import yaml
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs

from spectrseqtools.dataclasses import Sequence
from spectrseqtools.file_settings import load_alphabet
from spectrseqtools.parsers import MixturePostprocessingOptions


class EvaluationMetric:
    @property
    def dtype(self) -> pl.Struct:
        return pl.Struct(
            [
                pl.Field("predicted_string", pl.String),
                pl.Field("best_matching_target_string", pl.String),
                pl.Field("normalized_damerau_levenshtein_distance", pl.Float64),
                pl.Field("target_start_pos", pl.UInt64),
                pl.Field("target_end_pos", pl.UInt64),
                pl.Field("is_backward", pl.Boolean),
            ]
        )

    def score_query(self, query: Sequence, reference: Sequence) -> dict:
        fw_score, fw_idx = self.align(query=query, reference=reference)
        bw_score, bw_idx = self.align(query=query.reverse, reference=reference)

        if fw_score <= bw_score:
            res = {
                "predicted_string": query.to_str(),
                "best_matching_target_string": reference.to_str()[
                    fw_idx : fw_idx + len(query.sequence)
                ],
                "normalized_damerau_levenshtein_distance": fw_score,
                "target_start_pos": fw_idx,
                "target_end_pos": fw_idx + len(query.sequence),
                "is_backward": False,
            }

        else:
            res = {
                "predicted_string": query.to_str(),
                "best_matching_target_string": reference.to_str()[
                    bw_idx : bw_idx + len(query.sequence)
                ],
                "normalized_damerau_levenshtein_distance": bw_score,
                "target_start_pos": bw_idx,
                "target_end_pos": bw_idx + len(query.sequence),
                "is_backward": True,
            }
        return res

    @staticmethod
    def align(query: Sequence, reference: Sequence) -> Tuple[float, int]:
        pred_len = len(query.sequence)

        targets = [
            "".join(reference.sequence[idx : idx + pred_len])
            for idx in range(len(reference.sequence) - pred_len + 1)
        ]

        dist_list = normalized_damerau_levenshtein_distance_seqs(
            query.to_str(), targets
        )
        best_idx = np.argmin(dist_list)

        return dist_list[best_idx], best_idx


def evaluate_mixture(options: MixturePostprocessingOptions) -> None:
    """Evaluate predictions for complex mixture.

    Parameters
    ----------
    options : MixturePostprocessingOptions
        Options for prediction evaluation read by parser.

    """
    with open(options.meta, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
        if "true_sequence" not in meta:
            meta["true_sequence"] = "".join(meta["true_sequences"])

    # Load predicted sequences (if given)
    data = pl.read_csv(options.prediction, separator="\t")
    data = data.filter(pl.col("prediction").str.len_chars() > 0)
    data = data.rename(
        {
            "encoded_prediction": "predicted_sequence",
            "obs_mass": "intact_mass",
            "ms1_mass_group": "group_number",
            "adduct_type": "adduct_types",
        }
    )

    # Load true sequence
    masses = load_alphabet()
    target = Sequence.from_str(meta["true_sequence"]).to_encoding(masses)

    # Evaluate quality of predictions with metric
    metric = EvaluationMetric()
    df_align = data.with_columns(
        pl.col("prediction")
        .map_elements(
            lambda seq: metric.score_query(
                query=Sequence(Sequence.from_str(seq).to_encoding(masses)),
                reference=Sequence(sequence=target),
            ),
            return_dtype=metric.dtype,
        )
        .alias("results")
    ).unnest("results")

    # Filter out sequences with too high score
    df_align = df_align.filter(pl.col("normalized_damerau_levenshtein_distance") <= 1)

    # Reformat evaluated predictions
    df_align = df_align.rename({"group_number": "group", "adduct_types": "adduct_type"})
    df_align = df_align.select(
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
    )

    df_expanded_alignment = interpret_alignment_results(
        df_align, "".join(target), masses
    )

    df_align.write_csv(options.output_path / "df_alignment.csv", separator=",")
    df_expanded_alignment.write_csv(
        options.output_path / "df_expanded_alignment.csv", separator=","
    )
    return df_expanded_alignment


def interpret_alignment_results(df_alignment, target_sequence, masses):
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
