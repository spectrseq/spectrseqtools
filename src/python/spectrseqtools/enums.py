# -*- coding: utf-8 -*-
"""Module with enum classes."""

from enum import Enum


class AveragineBackbone(Enum):
    """Enum of backbone types for Averagine model used in deisotoping."""

    NONE = "no_backbone"
    PHOSPHATE = "phosphate"
    THIOPHOSPHATE = "thiophosphate"


class ErrorMetric(Enum):
    """Enum of types of metrics used for error calculations over multiple values."""

    L1NORM = "l1_norm"
    L2NORM = "l2_norm"


class LengthEstimatorMetric(Enum):
    """Enum of types of metrics used for sequence length estimation."""

    JACCARD = "jaccard"
    LP = "lp"


class SolverType(Enum):
    """Enum of types of solvers used for linear optimization."""

    CBC = "cbc"
    GUROBI = "gurobi"
    HIGHS = "highs"


class StartTag(Enum):
    """Enum of 5'-tags for sequences."""

    OH = "OH"
    P = "P"
    THIOAMIDE = "thioamide"
    BIOMERS_6FAM = "biomers_6fam"
    IDT_6FAM = "idt_6fam"


class EndTag(Enum):
    """Enum of 3'-tags for sequences."""

    OH = "OH"
    P = "P"
    CP = "CP"
    PCP_BIOTIN = "pcp_biotin"
    BIOMERS_BIOTIN_TEG = "biomers_biotin_teg"
    IDT_BIOTIN = "idt_biotin"
