import os
from pathlib import Path
import pytest
import polars as pl
import yaml

from itertools import product
from spectrseqtools.fragmentation_masses import (
    EndTag,
    FragmentationRelatedMasses,
    StartTag,
)


def get_fragmentation_mass_candidates(meta):
    start_tag = meta.get("5_prime_tag")
    end_tag = meta.get("3_prime_tag")

    # Both zeros indicate unspecified terminal chemistry.
    if start_tag == 0 and end_tag == 0:
        start_tag = end_tag = None

    start_tags = (
        (StartTag.OH, StartTag.P)
        if start_tag is None
        else (start_tag,)
    )
    end_tags = (
        (EndTag.OH, EndTag.CP, EndTag.P)
        if end_tag is None
        else (end_tag,)
    )

    candidates = []

    for start, end in product(start_tags, end_tags):
        candidates.append({
            "start_tag": str(getattr(start, "value", start)),
            "end_tag": str(getattr(end, "value", end)),
            "frag_ends": FragmentationRelatedMasses(
                start=float(start) if isinstance(start, (int, float)) else start,
                end=float(end) if isinstance(end, (int, float)) else end,
            ),
        })

    return candidates

@pytest.fixture(scope="module")
def raw_preprocessing_result():
    from spectrseqtools.multiplexing import preprocess_multiplexing
    from spectrseqtools.parsers import PreprocessingOptions

    testcasenumber = "test_05"

    testcase_dir = (
        Path(__file__).resolve().parent
        / "mixture_testcases"
        / testcasenumber
    )
    raw_path = testcase_dir / "fragments.raw"

    if not raw_path.is_file():
        pytest.skip("The RAW file for the test case is unavailable.")

    with open(testcase_dir / "fragments.meta.yaml", "r") as f:
            meta = yaml.safe_load(f)

    if meta.get("skip_preprocessing"):
        pytest.skip("Testcase is marked as skipped in meta.yaml")

    options = PreprocessingOptions(
        input=raw_path,
        meta=testcase_dir / "fragments.meta.yaml",
        output_dir=testcase_dir,
        singular_mode=False,
        min_time=None,
        max_time=None,
        window_size=0.2,
    )

    required_files = [
            testcase_dir / filename
            for filename in (
                "fragments.tsv",
                "fragments.singletons.tsv",
                "fragments.preprocessed.meta.yaml",
            )
        ]
    
    saved_outputs_available = all(
        path.is_file() and path.stat().st_size > 0
        for path in required_files
    )

    if saved_outputs_available:
        print(f"Using saved preprocessing outputs: {testcase_dir}")
    else:
        testcase_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running preprocessing; saving outputs to: {testcase_dir}")
        preprocess_multiplexing(options)

    fragments = pl.read_csv(
        testcase_dir / "fragments.tsv", separator="\t"
    )
    singletons = pl.read_csv(
        testcase_dir / "fragments.singletons.tsv", separator="\t"
    )
    meta = yaml.safe_load(
        (testcase_dir / "fragments.preprocessed.meta.yaml").read_text(
            encoding="utf-8"
        )
    )

    return {
        "fragments": fragments,
        "singletons": singletons,
        "meta": meta,
        "output_dir": testcase_dir,
        "options": options
    }

def test_raw_preprocessing_produces_outputs(raw_preprocessing_result):
    result = raw_preprocessing_result

    assert not result["fragments"].is_empty()
    assert not result["singletons"].is_empty()
    assert isinstance(result["meta"], dict)

    expected_files = (
        "fragments.tsv",
        "fragments.singletons.tsv",
        "fragments.preprocessed.meta.yaml"
    )

    for filename in expected_files:
        path = result["output_dir"] / filename 
        assert path.is_file(), f"Missing output: {filename}"
        assert path.stat().st_size > 0, f"Empty output: {filename}"

def test_fragment_and_singleton_groups_match(raw_preprocessing_result):
    result = raw_preprocessing_result
    
    fragments = result["fragments"]
    singletons = result["singletons"]

    fragment_groups = set(fragments["ms1_mass_group"].to_list())
    singleton_groups = set(singletons["ms1_mass_group"].to_list())

    assert None not in fragment_groups, "Fragments contain null group IDs."
    assert None not in singleton_groups, "Singletons contain null group IDs."

    assert fragment_groups == singleton_groups, (
        f"Groups missing from singletons: {fragment_groups - singleton_groups}; "
        f"groups missing from fragments: {singleton_groups - fragment_groups}"
    )


def test_exactly_one_intact_mass_per_group(raw_preprocessing_result):
    result = raw_preprocessing_result

    fragments = result["fragments"]

    assert fragments["is_ms1_mass"].null_count() == 0, (
        "Some fragment rows have no MS1/MS2 classification."
    )

    counts = fragments.group_by("ms1_mass_group").agg(
        pl.col("is_ms1_mass").sum().alias("intact_count")
    )
    invalid = counts.filter(pl.col("intact_count") != 1)

    assert invalid.is_empty(), (
        f"Groups without exactly one intact-mass row: {invalid.to_dicts()}"
    )


def test_each_group_has_ms2_fragments(raw_preprocessing_result):
    result = raw_preprocessing_result

    fragments = result["fragments"]

    counts = fragments.group_by("ms1_mass_group").agg(
        (~pl.col("is_ms1_mass")).sum().alias("ms2_count")
    )
    invalid = counts.filter(pl.col("ms2_count") < 1)

    assert invalid.is_empty(), (
        f"Groups without MS2 fragments: {invalid.to_dicts()}"
    )

def test_observed_masses_are_finite_and_positive(raw_preprocessing_result):
    result = raw_preprocessing_result

    fragments = result["fragments"]

    valid = (
        pl.col("observed_mass").is_finite()
        & (pl.col("observed_mass") > 0)
    )
    invalid = fragments.filter(~valid.fill_null(False))

    assert invalid.is_empty(), (
        f"Invalid observed masses: {invalid.head(10).to_dicts()}"
    )


def test_ms2_intensities_are_finite_and_positive(raw_preprocessing_result):
    result = raw_preprocessing_result
    
    fragments = result["fragments"]
    ms2 = fragments.filter(~pl.col("is_ms1_mass"))

    valid = (
        pl.col("intensity").is_finite()
        & (pl.col("intensity") > 0)
    )
    invalid = ms2.filter(~valid.fill_null(False))

    assert invalid.is_empty(), (
        f"Invalid MS2 intensities: {invalid.head(10).to_dicts()}"
    )


def test_ms2_masses_are_below_intact_mass(raw_preprocessing_result):
    result = raw_preprocessing_result
    
    fragments = result["fragments"]

    intact = fragments.filter(pl.col("is_ms1_mass")).select(
        "ms1_mass_group",
        pl.col("observed_mass").alias("intact_mass"),
    )
    ms2 = fragments.filter(~pl.col("is_ms1_mass"))

    comparison = ms2.join(
        intact,
        on="ms1_mass_group",
        how="left",
        validate="m:1",
    )

    valid = (
        pl.col("observed_mass").is_finite()
        & pl.col("intact_mass").is_finite()
        & (pl.col("observed_mass") < pl.col("intact_mass"))
    )
    invalid = comparison.filter(~valid.fill_null(False))

    assert invalid.is_empty(), (
        f"Fragments without a valid larger intact mass: "
        f"{invalid.head(10).to_dicts()}"
    )

def test_alphabet_ids_are_unique_per_group(raw_preprocessing_result):
    result = raw_preprocessing_result
    
    singletons = result["singletons"]

    assert singletons["id"].null_count() == 0, "Alphabet IDs contain nulls."

    blank = singletons.filter(pl.col("id").str.strip_chars() == "")
    assert blank.is_empty(), (
        f"Blank alphabet IDs: {blank.head(10).to_dicts()}"
    )

    duplicates = (
        singletons.group_by("ms1_mass_group", "id")
        .len()
        .filter(pl.col("len") > 1)
    )
    assert duplicates.is_empty(), (
        f"Duplicate alphabet entries: {duplicates.head(10).to_dicts()}"
    )


def test_alphabet_masses_are_finite_and_positive(raw_preprocessing_result):
    result = raw_preprocessing_result
    
    singletons = result["singletons"]

    valid = (
        pl.col("monoisotopic_mass").is_finite()
        & (pl.col("monoisotopic_mass") > 0)
    )
    invalid = singletons.filter(~valid.fill_null(False))

    assert invalid.is_empty(), (
        f"Invalid alphabet masses: {invalid.head(10).to_dicts()}"
    )


def test_intact_retention_intervals_are_valid(raw_preprocessing_result):
    result = raw_preprocessing_result
    
    fragments = result["fragments"]
    intact = fragments.filter(pl.col("is_ms1_mass"))

    valid = (
        pl.col("min_window_time").is_finite()
        & pl.col("max_window_time").is_finite()
        & (pl.col("min_window_time") <= pl.col("max_window_time"))
    )
    invalid = intact.filter(~valid.fill_null(False))

    assert invalid.is_empty(), (
        f"Invalid retention-time intervals: {invalid.head(10).to_dicts()}"
    )


def test_ms2_fragment_labels_are_unique_per_group(raw_preprocessing_result):
    result = raw_preprocessing_result
    
    fragments = result["fragments"]
    ms2 = fragments.filter(~pl.col("is_ms1_mass"))

    assert ms2["ppm_group"].null_count() == 0, (
        "Some MS2 fragments have no ppm_group label."
    )

    duplicates = (
        ms2.group_by("ms1_mass_group", "ppm_group")
        .len()
        .filter(pl.col("len") > 1)
    )
    assert duplicates.is_empty(), (
        f"Duplicate fragment labels: {duplicates.head(10).to_dicts()}"
    )

@pytest.fixture(scope="module")
def precursor_mass_metrics(raw_preprocessing_result):
    from spectrseqtools.dataclasses import Sequence
    from spectrseqtools.error_calculator import ErrorCalculator
    from spectrseqtools.file_settings import load_alphabet
    from spectrseqtools.fragmentation_masses import FragmentationRelatedMasses
    from spectrseqtools.nucleotide_alphabet import NucleotideAlphabet

    result = raw_preprocessing_result

    options = result["options"]

    meta = yaml.safe_load(
        options.meta.read_text(encoding="utf-8")
    )

    alphabet = load_alphabet(input_path=options.alphabet)
    assert alphabet["id"].n_unique() == alphabet.height, (
        "Reference alphabet must have unique IDs."
    )

    encoding_to_id = dict(
        zip(
            alphabet["encoding"].to_list(),
            alphabet["id"].to_list(),
        )
    )

    reference_alphabet = NucleotideAlphabet.from_dataframe(
        error=ErrorCalculator.with_metric(
            tolerance=options.tolerance,
            decimal_places=options.num_decimal_places,
        ),
        masses=alphabet,
    )
    alphabet_ids = {
        name 
        for nucleotide in reference_alphabet.alphabet
        for name in nucleotide.names
    }

    sequences = meta.get("true_sequences")
    if sequences is None:
        sequences = [meta["true_sequence"]]
    assert sequences, "No reference sequences were supplied in the metafile."

    if meta.get("in_silico_t1_digest"):
        assert len(sequences) == 1, "Input only one sequence with single-character encoding for in-silico digestion,"
        sequences = sequences[0].strip()
        cleavage_indices = [i+1 for i, b in enumerate(sequences) if b == 'G' and i < len(sequences)-1]
        boundaries = [0] + cleavage_indices + [len(sequences)]

        digested_seq = set()

        for mc in range(len(cleavage_indices) + 1):
            for i in range(len(boundaries) - 1 - mc):
                frag_seq = sequences[
                    boundaries[i]:boundaries[i+1+mc]
                ]

                digested_seq.add(
                    tuple(encoding_to_id[base] for base in frag_seq)
                )

        sequences = sorted(digested_seq)

        sequence_info = []

        for seq in sequences:
            sequence_mass = reference_alphabet.get_seq_weight(tuple(seq))

            sequence_info.append({
                "sequence": "".join(seq),
                "sequence_mass": sequence_mass
            })
    else:

        sequence_info = []

        for seq in sequences:
            sequence = Sequence.from_str(seq.strip()).sequence
            sequence_mass = reference_alphabet.get_seq_weight(tuple(sequence))

            sequence_info.append({
                "sequence": "".join(sequence),
                "sequence_mass": sequence_mass
            })

    fragment_ends = get_fragmentation_mass_candidates(meta)
    candidate_records = []
    for seq_info in sequence_info:
        for f_end in fragment_ends:
            frag_end = f_end["frag_ends"]

            candidate_records.append({
                "sequence": seq_info["sequence"],
                "sequence_mass": (
                    seq_info["sequence_mass"]
                    + frag_end.start
                    + frag_end.end
                ),
                "5_prime_end": f_end["start_tag"],
                "3_prime_end": f_end["end_tag"],
            })
    assert candidate_records, "No candidate sequence records were generated"
    candidates = pl.DataFrame(candidate_records)

    intact_masses = result["fragments"].filter(pl.col("is_ms1_mass"))
    assert not intact_masses.is_empty(), "No intact masses found."

    valid_masses = (
        pl.col("observed_mass").is_finite()
        & (pl.col("observed_mass") > 0)
    )
    assert intact_masses.filter(~valid_masses.fill_null(False)).is_empty(), (
        "Intact masses contain invalid values."
    )

    records = []

    for observed in intact_masses.iter_rows(named=True):

        ranked = candidates.with_columns(
            (
                (pl.lit(observed["observed_mass"]) - pl.col("sequence_mass"))
                / pl.col("sequence_mass")
            ).alias("mass_error")
        ).sort([
            pl.col("mass_error").abs(),
            pl.col("sequence"),
            pl.col("5_prime_end"),
            pl.col("3_prime_end"),
        ])

        best = ranked.row(0, named=True)

        records.append({
            "sequence": best["sequence"],
            "sequence_mass": best["sequence_mass"],
            "ppm_error": best["mass_error"] * 1e6,
            "within_tolerance": abs(best["mass_error"]) <= options.tolerance,
            "5_prime_end": best["5_prime_end"],
            "3_prime_end": best["3_prime_end"],
        })


    report = intact_masses.hstack(pl.DataFrame(records))
    report.write_csv(
        result["output_dir"] / "precursor_mass_metrics.tsv",
        separator="\t",
    )
    return report

def test_inspect_precursor_mass_metrics(precursor_mass_metrics):
    print(precursor_mass_metrics)