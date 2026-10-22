"""Dataset loaders for CALM experiments.

Each loader implements the `Dataset` protocol (fetch / preprocess / get_Xy /
get_feature_names). Classes are re-exported lazily (PEP 562) so that importing
`calm_paper.datasets` does not require the optional fetch dependencies
(ucimlrepo, openml, pmlb, aif360) unless the corresponding loader is used.
"""

from importlib import import_module

from .base import Dataset, DATA_ROOT

_EXPORTS = {
    "Adult": ".adult",
    "Ailerons": ".ailerons",
    "Airfoil": ".airfoil",
    "Bank": ".bank",
    "BikeSharing": ".bike_sharing",
    "CaliforniaHousing": ".california_housing",
    "CCPP": ".ccpp",
    "COMPAS": ".compas",
    "Electrical": ".electrical",
    "Elevators": ".elevators",
    "Energy": ".energy",
    "HELOC": ".heloc",
    "Magic": ".magic",
    "MIMIC2": ".mimic2",
    "No2": ".no2",
    "ParkinsonsTotal": ".parkinsons",
    "ParkinsonsMotor": ".parkinsons",
    "PMLB_APPENDICITIS": ".pmlb",
    "PMLB_PHONEME": ".pmlb",
    "PMLB_SPECTF": ".pmlb",
    "PMLB_CHURN": ".pmlb",
    "Sensory": ".sensory",
    "SeoulBike": ".seoul_bike",
    "SkillCraft": ".skill_craft",
    "Wine": ".wine",
}

# The 25 datasets of the paper's main tables, in the order the tables print them.
BENCHMARK = {
    "classification": [
        "Adult", "COMPAS", "HELOC", "MIMIC2", "PMLB_APPENDICITIS",
        "PMLB_PHONEME", "PMLB_SPECTF", "Magic", "Bank", "PMLB_CHURN",
    ],
    "regression": [
        "BikeSharing", "CaliforniaHousing", "ParkinsonsMotor", "ParkinsonsTotal",
        "SeoulBike", "Wine", "Energy", "CCPP", "Electrical", "Elevators",
        "No2", "Sensory", "Airfoil", "SkillCraft", "Ailerons",
    ],
}

# The explanatory example: three functions with a known structure.
SYNTHETIC = {"regression": ["Synthetic1", "Synthetic2", "Synthetic3"]}
_EXPORTS.update({name: ".synthetic" for name in SYNTHETIC["regression"]})

GROUPS = {"benchmark": BENCHMARK, "synthetic": SYNTHETIC}


def task_of(name: str) -> str:
    """`"classification"` or `"regression"` for a dataset name."""
    for group in GROUPS.values():
        for task, names in group.items():
            if name in names:
                return task
    raise KeyError(f"unknown dataset {name!r}")


__all__ = ["Dataset", "DATA_ROOT", "BENCHMARK", "SYNTHETIC", "GROUPS", "task_of"] + list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
