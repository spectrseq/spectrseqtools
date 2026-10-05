# -*- coding: utf-8 -*-
"""Plotting of coverage of true sequence by predicted sequences in mixture."""

import altair as alt
import polars as pl

from spectrseqtools.parsers import MixturePlottingOptions
from spectrseqtools.plotting import LEGEND_PARAMS, select_scale

STATUS_ORDER = ["match", "mismatch", "swap"]
STATUS_COLORS = {"match": "black", "mismatch": "red", "swap": "gold"}


def plot_coverage(options: MixturePlottingOptions) -> None:
    """Plot predicted sequences in mixture aligned to true sequence.

    Parameters
    ----------
    options : MixturePlottingOptions
        Options for mixture plot read by parser.

    """
    data = pl.read_csv(options.input, separator=",")
    chart = (
        alt.Chart(data)
        .mark_text()
        .encode(
            x=alt.X(
                "target_position:Q",
                title="Base position",
                scale=alt.Scale(
                    domain=[
                        data["target_position"].min(),
                        data["target_position"].max(),
                    ],
                    padding=10,
                ),
            ),
            y=alt.Y(
                "group:N",
                title="Sequence group",
                scale=alt.Scale(
                    padding=10,
                ),
            ),
            text="predicted_base:N",
            color=alt.Color(
                "status:N",
                scale=select_scale(order=STATUS_ORDER, colors=STATUS_COLORS),
                legend=alt.Legend(
                    **LEGEND_PARAMS,
                    orient="right",
                    title="",  # "Alignment status"
                ),
            ),
            tooltip=[
                "group",
                "target_position",
                "score",
                "predicted_base",
                "target_base",
                "status",
                "canonical_name",
                "intact_mass",
                "min_window_time",
                "max_window_time",
                "adduct_type",
            ],
        )
        .properties(
            width=10 * len(data.select("target_position").unique()),
            height=10 * len(data.select("group").unique()),
        )
    )

    chart.save(options.output_path)
