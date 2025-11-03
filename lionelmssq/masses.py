import importlib.resources
import polars as pl
from itertools import product

_COLS = ["nucleoside", "canonical_name", "monoisotopic_mass", "modification_rate"]


# TODO: Currently, the list of unmodified bases is only defined for RNA;
#  make it universally applicable
UNMODIFIED_BASES = ["A", "C", "G", "U"]


# Set number of binary-compressed masses per integer cell in DP table
COMPRESSION_RATE = 32


# Set the number of decimal places up to which to consider nucleoside masses
DECIMAL_PLACES = 5

# Set precision for calculations
# Note that for perfect matching, this should be equal or higher the precision
# to which nucleosides/sequences masses are reported, i.e. 1e-(DECIMAL_PLACES)
TOLERANCE = 1e-3

# Set relative matching threshold such that we consider
# abs(sum(masses)/target_mass - 1) < MATCHING_THRESHOLD for matching
# Note that the error is on the higher side than would be for a good
# calibrated machine (6ppm), but in the absence of an experimental measurement
# of this error, this conservative value works well
MATCHING_THRESHOLD = 10e-6


# Build dict with elemental masses
elements = pl.read_csv(
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv",
    separator="\t",
)
ELEMENT_MASSES = {
    row[elements.get_column_index("symbol")]: row[elements.get_column_index("mass")]
    for row in elements.iter_rows()
}

# Set mass for phosphate link between bases
PHOSPHATE_LINK_MASS = (
    ELEMENT_MASSES["P"] + 2 * ELEMENT_MASSES["O"] - ELEMENT_MASSES["H+"]
)


def initialize_nucleotide_df() -> pl.DataFrame:
    # Read nucleoside masses from file
    masses = pl.read_csv(
        (importlib.resources.files(__package__) / "assets" / "masses.tsv"),
        separator="\t",
    )
    assert masses.columns == _COLS

    # Round nucleoside masses
    masses = masses.with_columns(pl.col("monoisotopic_mass").round(DECIMAL_PLACES))

    # Group nucleosides by their mass, select a representative for each
    # group, and aggregate them into a list of equal-mass nucleosides
    masses = masses.group_by("monoisotopic_mass", maintain_order=True).agg(
        pl.col("nucleoside").first(),
        pl.col("nucleoside").unique().alias("nucleoside_list"),
        pl.col("modification_rate").max(),
    )

    # Convert nucleosides to nucleotides and add new columns for theoretical
    # m/z values and integer masses for the DP algorithm
    masses = masses.with_columns(
        pl.col("monoisotopic_mass")
        .add(
            PHOSPHATE_LINK_MASS - ELEMENT_MASSES["H+"]
        )  # Subtract one proton from nucleotide (for singleton charge)
        .alias("theoretical_mz")
    ).with_columns(
        ((pl.col("monoisotopic_mass") + PHOSPHATE_LINK_MASS) / TOLERANCE)
        .round(0)
        .cast(pl.Int64)
        .alias("tolerated_integer_masses")
    )

    return masses


EXPLANATION_MASSES = initialize_nucleotide_df()


# METHOD: Precompute all weight changes caused by breakages and adapt the
# target masses accordingly while finding compositions explaining it.
# We consider tags at the 5'- or 3'-end to be possible breakage options.


def build_breakage_dict(mass_5_prime, mass_3_prime):
    element_masses = ELEMENT_MASSES

    # Initialize dict with masses for 5'-end of fragments
    start_dict = {
        # Remove O from SU and add START tag (without H)
        "START": mass_5_prime - element_masses["O"] - element_masses["H+"],
        # Add H to SU to achieve neutral charge
        "c/y": element_masses["H+"],
    }

    # Initialize dict with masses for 3'-end of fragments
    end_dict = {
        # Remove PO3H from SU and add END tag (without H)
        "END": mass_3_prime
        - element_masses["P"]
        - 3 * element_masses["O"]
        - 2 * element_masses["H+"],
        # Remove H from SU to achieve neutral charge
        "c/y": -element_masses["H+"],
    }

    # Collect all unique breakage-related mass combinations in dict
    breakage_dict = {}
    for start, end in list(product(start_dict.keys(), end_dict.keys())):
        val = int((start_dict[start] + end_dict[end]) / TOLERANCE)
        if val not in breakage_dict:
            breakage_dict[val] = []
        breakage_dict[val] += [f"{start}_{end}"]

    return breakage_dict
