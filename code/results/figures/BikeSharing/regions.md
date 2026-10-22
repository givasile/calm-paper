# BikeSharing — CALM, depth 2, fitted on all 17379 rows

7 conditional interactions, 3 of 11 features partitioned

- **hr**: workingday = 0, temp < 0.46 (2789 rows); workingday = 0, temp ≥ 0.46 (2725 rows); workingday = 1, yr = 0 (5911 rows); workingday = 1, yr = 1 (5954 rows); cut by other features at [np.float64(6.5), np.float64(9.5), np.float64(14.5)]
- **temp**: hr ≤ 6, mnth ≤ 4 (1603 rows); hr ≤ 6, mnth ≥ 5 (3398 rows); 7 ≤ hr ≤ 14 (5821 rows); hr ≥ 15 (6557 rows); cut by other features at [0.461, 0.559]
- **hum**: hr ≤ 6 (5001 rows); 7 ≤ hr ≤ 9 (2181 rows); hr ≥ 10, temp < 0.56 (5339 rows); hr ≥ 10, temp ≥ 0.56 (4858 rows)
