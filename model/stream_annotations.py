"""Authoritative change annotations for the local INSECTS benchmark files.

Primary source:
Souza et al. (2020), "Challenges in Benchmarking Stream Learning Algorithms
with Real-world Data", Data Mining and Knowledge Discovery 34, 1805–1858.
https://doi.org/10.1007/s10618-020-00698-5

The paper distinguishes exact abrupt boundaries from streams whose changes are
gradual or incremental. Reference points in the latter must not be scored as
instantaneous ground-truth drift alarms.
"""

from pathlib import Path
from typing import Dict, Optional


SOURCE = {
    "citation": (
        "Souza, V.M.A., dos Reis, D.M., Maletzke, A.G. et al. "
        "Challenges in benchmarking stream learning algorithms with real-world data. "
        "Data Mining and Knowledge Discovery 34, 1805–1858 (2020)."
    ),
    "doi": "https://doi.org/10.1007/s10618-020-00698-5",
    "preprint": "https://arxiv.org/abs/2005.00113",
}


INSECTS_ANNOTATIONS: Dict[str, dict] = {
    "INSECTS_abrupt_balanced.csv": {
        "instances": 52848,
        "change_pattern": "abrupt",
        "exact_abrupt_points": [14352, 19500, 33240, 38682, 39510],
        "reference_points": [14352, 19500, 33240, 38682, 39510],
    },
    "INSECTS_abrupt_imbalanced.csv": {
        "instances": 355275,
        "change_pattern": "abrupt",
        "exact_abrupt_points": [83859, 128651, 182320, 242883, 268380],
        "reference_points": [83859, 128651, 182320, 242883, 268380],
    },
    "INSECTS_gradual_balanced.csv": {
        "instances": 24150,
        "change_pattern": "incremental_gradual",
        "exact_abrupt_points": [],
        "reference_points": [14028],
        "annotation_warning": "The reference marks a gradual/incremental change, not an instantaneous alarm target.",
    },
    "INSECTS_gradual_imbalanced.csv": {
        "instances": 143323,
        "change_pattern": "incremental_gradual",
        "exact_abrupt_points": [],
        "reference_points": [58159],
        "annotation_warning": "The reference marks a gradual/incremental change, not an instantaneous alarm target.",
    },
    "INSECTS_incremental_balanced.csv": {
        "instances": 57018,
        "change_pattern": "incremental_throughout",
        "exact_abrupt_points": [],
        "reference_points": [],
        "annotation_warning": "Incremental evolution occurs throughout the stream.",
    },
    "INSECTS_incremental_imbalanced.csv": {
        "instances": 452044,
        "change_pattern": "incremental_throughout",
        "exact_abrupt_points": [],
        "reference_points": [],
        "annotation_warning": "Incremental evolution occurs throughout the stream.",
    },
    "INSECTS_incremental_abrupt_balanced.csv": {
        "instances": 79986,
        "change_pattern": "incremental_with_abrupt_recurrence",
        "exact_abrupt_points": [26568, 53364],
        "reference_points": [26568, 53364],
        "annotation_warning": "Incremental changes also occur between the exact abrupt recurrence boundaries.",
    },
    "INSECTS_incremental_reoccurring_balanced.csv": {
        "instances": 79986,
        "change_pattern": "incremental_reoccurring",
        "exact_abrupt_points": [],
        "reference_points": [26568, 53364],
        "annotation_warning": "Cycle boundaries are references, not abrupt ground-truth alarms.",
    },
}


def get_stream_annotation(csv_path: str) -> Optional[dict]:
    """Return a copy of the annotation for a known local dataset."""
    annotation = INSECTS_ANNOTATIONS.get(Path(csv_path).name)
    if annotation is None:
        return None
    return {**annotation, "source": dict(SOURCE)}

