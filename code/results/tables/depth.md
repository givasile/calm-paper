## regression — gap closed, 12 of 14 datasets

| depth | datasets | median | quartiles | half closed | median pairs, all datasets |
|---|---|---|---|---|---|
| 1 | 12 | 17% | 2% – 42% | 2 | 1.9 |
| 2 | 12 | 37% | 8% – 55% | 4 | 4.5 |
| 3 | 12 | 49% | 13% – 64% | 6 | 9.6 |

no gap to close: Sensory, SkillCraft

### BikeSharing — depth ladder (rmse)

| depth | rmse | pairs | gap closed |
|---|---|---|---|
| GAM | 100.2 ± 0.99 |  | 0% |
| 1 | 73.74 ± 0.96 | 1.8 | 43% |
| 2 | 62.4 ± 2 | 4.6 | 62% |
| 3 | 56.71 ± 1.3 | 3.2 | 71% |
| 4 | 52.54 ± 1.5 | 5.0 | 78% |
| 5 | 49.68 ± 1.3 | 8.0 | 83% |
| 6 | 48.67 ± 1.6 | 9.8 | 85% |
| XGB | 39.35 ± 1.4 |  | 100% |

## classification — gap closed, 5 of 10 datasets

| depth | datasets | median | quartiles | half closed | median pairs, all datasets |
|---|---|---|---|---|---|
| 1 | 5 | 17% | 10% – 17% | 1 | 2.5 |
| 2 | 5 | 31% | 27% – 36% | 1 | 6.9 |
| 3 | 5 | 41% | 35% – 44% | 1 | 10.8 |

no gap to close: Adult, COMPAS, HELOC, PMLB_APPENDICITIS, PMLB_SPECTF

### Adult — depth ladder (accuracy)

| depth | accuracy | pairs | gap closed |
|---|---|---|---|
| GAM | 0.8699 ± 0.0016 |  | – |
| 1 | 0.8702 ± 0.0018 | 4.0 | – |
| 2 | 0.8699 ± 0.0015 | 9.2 | – |
| 3 | 0.8699 ± 0.0026 | 11.8 | – |
| 4 | 0.8691 ± 0.003 | 12.6 | – |
| 5 | 0.8693 ± 0.002 | 16.2 | – |
| 6 | 0.8679 ± 0.0014 | 20.2 | – |
| XGB | 0.87 ± 0.0025 |  | – |

### Bank — depth ladder (accuracy)

| depth | accuracy | pairs | gap closed |
|---|---|---|---|
| GAM | 0.9027 ± 0.002 |  | 0% |
| 1 | 0.904 ± 0.0028 | 3.2 | 17% |
| 2 | 0.9054 ± 0.0023 | 7.6 | 36% |
| 3 | 0.906 ± 0.0028 | 12.2 | 44% |
| 4 | 0.9068 ± 0.002 | 21.0 | 55% |
| 5 | 0.9067 ± 0.0015 | 24.8 | 53% |
| 6 | 0.9075 ± 0.0017 | 31.0 | 64% |
| XGB | 0.9101 ± 0.0037 |  | 100% |

### PMLB_PHONEME — depth ladder (accuracy)

| depth | accuracy | pairs | gap closed |
|---|---|---|---|
| GAM | 0.8266 ± 0.0071 |  | 0% |
| 1 | 0.8338 ± 0.011 | 2.4 | 10% |
| 2 | 0.8494 ± 0.0055 | 5.4 | 31% |
| 3 | 0.8568 ± 0.0091 | 6.2 | 41% |
| 4 | 0.8583 ± 0.012 | 7.8 | 43% |
| 5 | 0.8595 ± 0.011 | 8.0 | 45% |
| 6 | 0.8703 ± 0.005 | 11.2 | 59% |
| XGB | 0.9003 ± 0.0037 |  | 100% |

