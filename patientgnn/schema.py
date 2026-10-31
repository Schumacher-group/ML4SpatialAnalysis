"""Assay schema and feature definitions used by the patient pipeline."""

CACHE_VERSION = "patient-roi-v2-qc-20260722"

ALL_MARKERS = [
    "Alpha-SMA", "B7-H4", "Beta-Catenin", "CD107a", "CD11b", "CD14",
    "CD16", "CD163", "CD20", "CD27", "CD3", "CD31", "CD366", "CD38",
    "CD4", "CD44", "CD45", "CD45RO", "CD68", "CD8a", "Carboplatin",
    "Collage-Type_I", "DNA1", "DNA2", "E-Cadherin", "EGFR", "FOXP3",
    "Granzyme-B", "HLA-DR-DQ-DP", "Ki-67", "PD-1", "PD-L1", "PD-L2",
    "Pan-keratin", "Tbet", "VEGF", "Vimentin", "p53",
]

# DNA channels and Carboplatin are acquisition/QC channels, not biological
# response features.  They are retained in the raw data audit but excluded
# from model inputs by default to reduce technical shortcut learning.
TECHNICAL_MARKERS = ["Carboplatin", "DNA1", "DNA2"]
DEFAULT_MARKERS = [m for m in ALL_MARKERS if m not in TECHNICAL_MARKERS]

# Compact morphology set. Morphology is cached for future controlled
# ablations, but is deliberately not part of the three primary feature-mode
# comparisons (expressions, celltypes, combined).
MORPHOLOGY_FEATURES = [
    "cell_size",
    "area",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "major_minor_axis_ratio",
]

LABEL_MAP = {"Non-Responder": 0, "Responder": 1, "pCR": 1}

