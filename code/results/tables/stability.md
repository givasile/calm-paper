### Folds — 5 refits, each on 80 % of the rows

25 datasets, 389 features, refits per dataset [5]

| level | 5 / 5 | >= 4 / 5 | pairwise |
|---|---|---|---|
| 1 — is the feature split at all | 78 % | 91 % | 89 % |
| 2 — same root conditioner | 75 % | 87 % | 86 % |
| 3 — same full structure | 72 % | 82 % | 81 % |

**Threshold spread** (rules recurring in >= 3 refits, 88 rules): mean 0.135 σ, median 0.081 σ, worst 0.83 σ; 73 % within 0.2 σ, 56 % within 0.1 σ.

### Bootstrap — 20 refits, each on n rows drawn with replacement

25 datasets, 389 features, refits per dataset [20]

| level | 5 / 5 | >= 4 / 5 | pairwise |
|---|---|---|---|
| 1 — is the feature split at all | 75 % | 90 % | 88 % |
| 2 — same root conditioner | 72 % | 86 % | 84 % |
| 3 — same full structure | 69 % | 81 % | 80 % |

5 / 5 and >= 4 / 5 are averages over 4 disjoint groups of 5 refits.

**Threshold spread** (rules recurring in >= 12 refits, 48 rules): mean 0.163 σ, median 0.097 σ, worst 0.95 σ; 75 % within 0.2 σ, 56 % within 0.1 σ.

