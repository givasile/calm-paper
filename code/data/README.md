# Datasets

The 25 benchmark datasets of the paper's main tables. Most are fetched by
their loader on first use; three are committed here; two cannot be
redistributed and must be obtained by you.

`MANIFEST.json` holds the shape and a sha256 of every dataset as the loaders
return it, so a copy you obtained can be checked against the one we used.

## Obtain yourself

| dataset | why not here | how to get it |
|---|---|---|
| HELOC | FICO's terms forbid redistribution | download from the FICO Explainable Machine Learning Challenge (community.fico.com) and save it as `data/heloc_dataset_v1 (1).csv` |
| MIMIC2 | MIMIC-II is under the PhysioNet data use agreement | obtain credentialed access on PhysioNet, then the preprocessed split used by NODE-GAM (`nodegam.data.fetch_MIMIC2`): put `mimic2.data`, `train0-4.txt` and `test0-4.txt` in `data/calm-data/mimic2/` |

Without them, their cells fail with a pointer to this file; everything else
runs.

## Committed here

| dataset | file | source | licence |
|---|---|---|---|
| BikeSharing | `Bike-Sharing-Dataset/` | UCI 275 | CC BY 4.0; cite Fanaee-T and Gama (2013), see its `Readme.txt` |
| Energy | `energydata_complete.csv` | UCI 374 (Appliances Energy Prediction) | CC BY 4.0 |
| SkillCraft | `SkillCraft1_Dataset.csv` | UCI 272 | CC BY 4.0 |

## Fetched by the loaders

| dataset | source |
|---|---|
| Adult | UCI 2, via `ucimlrepo` |
| Bank | UCI 222, via `ucimlrepo` |
| CCPP | UCI 294, via `ucimlrepo` |
| Magic | UCI 159, via `ucimlrepo` |
| ParkinsonsMotor, ParkinsonsTotal | UCI 189, via `ucimlrepo` |
| SeoulBike | UCI 560, via `ucimlrepo` |
| Wine | UCI 186, via `ucimlrepo` |
| COMPAS | ProPublica, via `aif360` |
| CaliforniaHousing | `sklearn.datasets.fetch_california_housing` |
| PMLB_APPENDICITIS, PMLB_PHONEME, PMLB_SPECTF, PMLB_CHURN | PMLB, via `pmlb` |
| Ailerons | OpenML 296 |
| Elevators | OpenML 216 |
| No2 | OpenML 547 |
| Sensory | OpenML 826 |
| Airfoil | OpenML 43919 |
| Electrical | OpenML 43007 (run, but dropped from every table: target leak) |

Each keeps the licence of its source.
