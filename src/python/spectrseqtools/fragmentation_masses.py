# -*- coding: utf-8 -*-
"""Module for fragment-related masses (outside SUs)."""

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import polars as pl
import yaml

from spectrseqtools.enums import EndTag, StartTag

DEFAULT_ELEMENT_PATH = (
    importlib.resources.files(__package__) / "assets" / "element_masses.tsv"
)

# Build dict with elemental masses
elements = pl.read_csv(DEFAULT_ELEMENT_PATH, separator="\t")
ELEMENT_MASSES = {
    row[elements.get_column_index("symbol")]: row[elements.get_column_index("mass")]
    for row in elements.iter_rows()
}


@dataclass
class FragmentationRelatedMasses:
    """Class for fragmentation masses relative to standard units."""

    start: float = "idt_6fam"
    end: float = "idt_biotin"

    def __post_init__(self):
        # Remove O from SU and add START tag (-H) for 5'-end of terminal fragments
        self.start = round(
            select_start_tag(self.start) - (ELEMENT_MASSES["O"] + ELEMENT_MASSES["H+"]),
            10,
        )
        # Remove PO3H from SU and add END tag (-H) for 3'-end of terminal fragments
        self.end = round(
            select_end_tag(self.end)
            - (
                ELEMENT_MASSES["P"] + 3 * ELEMENT_MASSES["O"] + 2 * ELEMENT_MASSES["H+"]
            ),
            10,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        """Initialize from meta file."""
        with open(file_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        return cls(
            start=select_start_tag(meta.setdefault("5_prime_tag", StartTag.IDT_6FAM)),
            end=select_end_tag(meta.setdefault("3_prime_tag", EndTag.IDT_BIOTIN)),
        )

    @property
    def a_w(self) -> float:
        """Return mass needed to convert SUs to a/w-fragmentation (5'-direction).

        Add PO3H2 to SU to achieve neutral charge and a/w-fragmentation.
        """
        return ELEMENT_MASSES["P"] + 3 * ELEMENT_MASSES["O"] + 2 * ELEMENT_MASSES["H+"]

    @property
    def b_x(self) -> float:
        """Return mass needed to convert SUs to b/x-fragmentation (5'-direction).

        Add P2O to SU to achieve neutral charge and b/x-fragmentation.
        """
        return ELEMENT_MASSES["P"] + 2 * ELEMENT_MASSES["O"]

    @property
    def c_y(self) -> float:
        """Return mass needed to convert SUs to c/y-fragmentation (5'-direction).

        Add H to SU to achieve neutral charge and c/y-fragmentation.
        """
        return ELEMENT_MASSES["H+"]

    @property
    def d_z(self) -> float:
        """Return mass needed to convert SUs to d/z-fragmentation (5'-direction).

        Remove OH from SU to achieve neutral charge and d/z-fragmentation.
        """
        return -(ELEMENT_MASSES["O"] + ELEMENT_MASSES["H+"])

    @property
    def phosphate_link(self) -> float:
        """Return phosphate link mass needed to convert nucleoside to nucleotide SU."""
        return ELEMENT_MASSES["P"] + 2 * ELEMENT_MASSES["O"] - ELEMENT_MASSES["H+"]

    @property
    def tag_mass(self) -> float:
        """Return mass of empty sequence only consisting of tags."""
        return round(self.start + self.end + self.a_w - self.d_z, 10)


def select_start_tag(tag: StartTag | float) -> float:
    """Select mass for 5'-tag."""
    # Return float value directly
    if isinstance(tag, float):
        return tag

    # Convert string into Enum object
    if isinstance(tag, str):
        tag = StartTag(tag)

    # Select mass matching tag
    match tag:
        case StartTag.OH:
            return 18.009482
        case StartTag.P:
            return 97.975308
        case StartTag.THIOAMIDE:
            return 170.98
        case StartTag.IDT_6FAM:
            return 555.1294
        case StartTag.BIOMERS_6FAM:
            return 555.1294
        case _:
            raise NotImplementedError(f"Support for '{tag}' is currently not given.")


def select_end_tag(tag: EndTag | float) -> float:
    """Select mass for 3'-tag."""
    # Return float value directly
    if isinstance(tag, float):
        return tag

    # Convert string into Enum object
    if isinstance(tag, str):
        tag = EndTag(tag)

    # Select mass matching tag
    match tag:
        case EndTag.OH:
            return 18.009482
        case EndTag.CP:
            return 79.965826
        case EndTag.P:
            return 97.975308
        case EndTag.PCP_BIOTIN:
            return 728.2
        case EndTag.IDT_BIOTIN:
            return 455.1491
        case EndTag.BIOMERS_BIOTIN_TEG:
            return 587.2278
        case _:
            raise NotImplementedError(f"Support for '{tag}' is currently not given.")
