# -*- coding: utf-8 -*-
"""Classification of fragments."""

from itertools import product
from pathlib import Path

from spectrseqtools.error_calculator import ErrorCalculator
from spectrseqtools.fragmentation_masses import FragmentationRelatedMasses
from spectrseqtools.fragments import RawFragments, StandardUnitFragments

# METHOD: Precompute all weight changes caused by fragmentation and adapt the
# target masses accordingly while finding compositions explaining it.
# We consider tags at the 5'- or 3'-end to be possible fragmentation options.
# During classification of a given fragment, for each fragmentation option that
# yields a valid mass (i.e. one that has any valid composition),
# duplicate the fragment and determine its fragmentation-independent standard-unit
# mass by subtracting the weight imposed by the fragmentation.


class FragmentClassifier:
    """Class to classify fragments."""

    def __init__(
        self,
        file_path: Path,
        error: ErrorCalculator,
        reduced: bool = True,
    ):
        """
        Initialize classifier by building dictionary over fragmentation options.

        Parameters
        ----------
        file_path : Path
            Path to meta file.
        error : ErrorCalculator
            Error Calculator.
        reduced : bool
            Flag whether reduced fragmentation list (i.e. only c/y) is used.

        """
        self.error = error

        # Initialize fragmentation-related masses
        frag_masses = FragmentationRelatedMasses.from_file(file_path=file_path)

        # Initialize dicts with masses for 5'- and 3'-end of fragments
        start_dict = {"START": frag_masses.start, "c/y": frag_masses.c_y}
        end_dict = {"END": frag_masses.end, "c/y": -frag_masses.c_y}

        # Add a/w-, b/x-, and d/z-fragmentation for full dict version
        if not reduced:
            start_dict["a/w"] = frag_masses.a_w
            start_dict["b/x"] = frag_masses.b_x
            start_dict["d/z"] = frag_masses.d_z

            end_dict["a/w"] = -frag_masses.a_w
            end_dict["b/x"] = -frag_masses.b_x
            end_dict["d/z"] = -frag_masses.d_z

        # Collect all unique fragmentation-related mass combinations in dict
        fragmentation_dict = {}
        for start, end in list(product(start_dict, end_dict)):
            val = int((start_dict[start] + end_dict[end]) / error.precision)
            if val not in fragmentation_dict:
                fragmentation_dict[val] = []
            fragmentation_dict[val] += [f"{start}_{end}"]

        self.fragmentation_options = fragmentation_dict

    @property
    def start_end_fragmentation(self) -> int:
        """Return integer mass of 'START_END' fragmentation."""
        return [
            mass
            for mass, fragmentation in self.fragmentation_options.items()
            if "START_END" in fragmentation
        ][0] * self.error.precision

    def classify(self, fragments: RawFragments) -> StandardUnitFragments:
        """
        Classify raw fragments while standardizing them.

        Parameters
        ----------
        fragments : RawFragments
            Raw fragments.

        Returns
        -------
        StandardUnitFragments
            SU-fragments.

        """
        return fragments.standardize(
            fragmentation_dict=self.fragmentation_options, error=self.error
        )
