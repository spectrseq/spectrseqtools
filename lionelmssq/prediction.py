from dataclasses import dataclass
from pathlib import Path
from typing import List, Self, Set
import polars as pl
from loguru import logger

from lionelmssq.common import calculate_error_threshold, calculate_explanations
from lionelmssq.linear_program import LinearProgramInstance
from lionelmssq.mass_table import DynamicProgrammingTable
from lionelmssq.masses import PHOSPHATE_LINK_MASS
from lionelmssq.skeleton_building import SkeletonBuilder


@dataclass
class Prediction:
    sequence: List[str]
    fragments: pl.DataFrame

    @classmethod
    def from_files(cls, sequence_path: Path, fragments_path: Path) -> Self:
        with open(sequence_path) as f:
            head, seq = f.readlines()
            assert head.startswith(">")

        fragments = pl.read_csv(fragments_path, separator="\t")
        return Prediction(sequence=seq.strip(), fragments=fragments)


class Predictor:
    def __init__(
        self,
        dp_table: DynamicProgrammingTable,
        explanation_masses: pl.DataFrame,
    ):
        self.explanation_masses = explanation_masses
        self.dp_table = dp_table

    def predict(
        self,
        fragments: pl.DataFrame,
        solver_params: dict,
    ) -> Prediction:
        fragments = (
            fragments.with_row_index(name="orig_index")
            .sort("standard_unit_mass")
            .with_row_index(name="index")
        )
        print(len(fragments))

        fragments = fragments.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("min_end"),
            pl.lit(-1, dtype=pl.Int64).alias("max_end"),
        )

        # Collect internal fragments
        frag_internal = fragments.filter(
            ~pl.col("breakage").str.contains("START")
            & ~pl.col("breakage").str.contains("END")
        )

        # Roughly explain the mass differences (to reduce the alphabet)
        # Note there may be faulty mass fragments leading to not truly existent values
        explanations = self.collect_diff_explanations_for_su(fragments=fragments)

        # TODO: Also consider that the observations are not complete and that
        #  we probably don't see all the letters as diffs or singletons.
        #  Hence, maybe do the following: Solve first with the reduced
        #  alphabet, and if the optimization does not yield a sufficiently
        #  good result, then try again with an extended alphabet.

        # Reduce nucleotide alphabet based on fragments
        observed_nucleotides = {
            nuc
            for expls in explanations.values()
            if expls is not None
            for expl in expls
            for nuc in expl
        }
        _ = self._reduce_alphabet(observed_nucleotides)

        skeleton_builder = SkeletonBuilder(
            explanations=explanations,
            dp_table=self.dp_table,
        )

        # Build skeleton sequence from both sides and align them into final sequence
        skeleton_seq, frag_terminal = skeleton_builder.build_skeleton(fragments)

        # Reduce nucleotide alphabet based on skeleton
        nucleotides = {nuc for skeleton_pos in skeleton_seq for nuc in skeleton_pos}
        _ = self._reduce_alphabet(nucleotides)

        # Remove all "internal" fragment duplicates that are truly terminal fragments
        frag_internal = frag_internal.filter(
            ~pl.col("fragment_index").is_in(
                frag_terminal.get_column("fragment_index").to_list()
            )
        )

        # Rebuild fragment dataframe from internal and terminal fragments
        fragments = frag_internal.vstack(frag_terminal).sort("index")

        # Filter out all internal fragments that do not fit anywhere in skeleton
        print(
            "Number of internal fragments before filtering: ",
            len(
                fragments.filter(
                    ~pl.col("breakage").str.contains("START")
                    & ~pl.col("breakage").str.contains("END")
                )
            ),
        )
        fragments = self.filter_with_lp(
            fragments=fragments,
            skeleton_seq=skeleton_seq,
            solver_params=solver_params,
        )
        print(
            "Number of internal fragments after filtering: ",
            len(
                fragments.filter(
                    ~pl.col("breakage").str.contains("START")
                    & ~pl.col("breakage").str.contains("END")
                )
            ),
        )

        print("Fragments considered for fitting, n_fragments = ", len(fragments))

        if len(fragments.filter(pl.col("breakage").str.contains("START"))) == 0:
            logger.warning(
                "No start fragments provided, this will likely lead to suboptimal results."
            )

        if len(fragments.filter(pl.col("breakage").str.contains("END"))) == 0:
            logger.warning(
                "No end fragments provided, this will likely lead to suboptimal results."
            )

        lp_instance = LinearProgramInstance(
            fragments=fragments,
            dp_table=self.dp_table,
            skeleton_seq=skeleton_seq,
        )

        return Prediction(*lp_instance.evaluate(solver_params))

    def _reduce_alphabet(self, nucleotide_list: Set[str]) -> pl.DataFrame:
        reduced = self.explanation_masses.filter(
            pl.col("nucleoside").is_in(nucleotide_list)
        )
        reduced = reduced.with_columns(
            (pl.col("monoisotopic_mass") + PHOSPHATE_LINK_MASS).alias(
                "standard_unit_mass"
            )
        )
        print("Nucleosides considered for fitting after alphabet reduction:", reduced)

        self.dp_table.adapt_individual_modification_rates_by_alphabet_reduction(
            nucleotide_list
        )

        return reduced

    def filter_with_lp(
        self,
        fragments: pl.DataFrame,
        skeleton_seq: list,
        solver_params: dict,
    ) -> pl.DataFrame:
        is_invalid = []
        for idx in range(len(fragments)):
            # Skip terminal (i.e. non-internal) fragments
            if ("START" in fragments.item(idx, "breakage")) or (
                "END" in fragments.item(idx, "breakage")
            ):
                continue

            # Initialize LP instance for a singular fragment
            filter_instance = LinearProgramInstance(
                fragments=fragments[idx],
                dp_table=self.dp_table,
                skeleton_seq=skeleton_seq,
            )

            # Check whether fragment can feasibly be aligned to skeleton
            if not filter_instance.check_feasibility(
                solver_params=solver_params,
                threshold=self.dp_table.tolerance
                * fragments.item(idx, "observed_mass"),
            ):
                is_invalid.append(fragments.item(idx, "index"))

        # Return only valid fragments
        return fragments.filter(~pl.col("index").is_in(is_invalid))

    def collect_diff_explanations_for_su(self, fragments: pl.DataFrame) -> dict:
        # Collect explanation for all reasonable mass differences for each side
        explanations = {
            **self.collect_explanations_per_side(
                fragments=fragments.filter(pl.col("breakage").str.contains("START")),
            ),
            **self.collect_explanations_per_side(
                fragments=fragments.filter(pl.col("breakage").str.contains("END")),
            ),
        }

        # Collect singleton masses
        singleton_list = fragments.filter(pl.col("is_singleton"))

        idx_observed_mass = fragments.get_column_index("observed_mass")
        idx_su_mass = fragments.get_column_index("standard_unit_mass")
        for singleton in singleton_list.rows():
            explanations[singleton[idx_su_mass]] = calculate_explanations(
                diff=singleton[idx_su_mass],
                threshold=self.dp_table.tolerance * singleton[idx_observed_mass],
                dp_table=self.dp_table,
            )

        return explanations

    def collect_explanations_per_side(self, fragments: pl.DataFrame) -> dict:
        max_weight = (
            max(self.explanation_masses.get_column("monoisotopic_mass").to_list())
            + PHOSPHATE_LINK_MASS
        )
        su_masses = fragments.get_column("standard_unit_mass").to_list()
        observed_masses = fragments.get_column("observed_mass").to_list()
        start = 0
        end = 1

        explanations = {}
        while end < len(fragments):
            # Skip singletons
            if (end - start) <= 0:
                end += 1
                continue

            # Determine mass difference between fragments
            diff = su_masses[end] - su_masses[start]

            # If mass difference > any nucleotide mass, drop 1st fragment in window
            if diff > max_weight:
                start += 1
                end = start + 1
                continue

            diff_error = calculate_error_threshold(
                observed_masses[start],
                observed_masses[end],
                self.dp_table.tolerance,
            )
            expl = calculate_explanations(
                diff=diff,
                threshold=diff_error,
                dp_table=self.dp_table,
            )
            if expl is not None and len(expl) >= 1:
                explanations[diff] = expl
            if end == len(fragments) - 1:
                start += 1
            else:
                end += 1

        return explanations
