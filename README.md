# calm-paper

The code behind *Interpretability-by-Design with Accurate Locally Additive
Models and Conditional Feature Effects* (CALM, NeurIPS 2026).

[Paper](https://arxiv.org/abs/2602.16503) ·
[Package](https://github.com/givasile/calm-additive) ·
[Page](https://givasile.github.io/calm-additive)

CALM itself is the package `calm-additive` (`pip install calm-additive`).
This repository holds what the paper needs and a user does not: the
datasets, the evaluation protocol, the baselines, the experiments, their
results, and the scripts that turn them into the paper's tables and figures.

| folder | what it is |
|---|---|
| [`code/`](code/) | everything that runs, and its results — start at [`code/README.md`](code/README.md) |
| [`paper/`](paper/) | the paper's LaTeX source and [PDF](paper/main.pdf); its tables (`paper/generated/`) and figures come from `code/` |
| [`poster/`](poster/) | the NeurIPS 2026 poster: LaTeX source and [PDF](poster/main.pdf); its figures are the paper's |

```bash
cd code
uv sync                            # the exact environment
uv run python reports/tables.py --out ../paper/generated   # the paper's tables, from the committed results
cd ../paper && latexmk -pdf main.tex                         # the paper
```

## Citation

```bibtex
@inproceedings{gkolemis2026calm,
  title     = {Interpretability-by-Design with Accurate Locally Additive Models
               and Conditional Feature Effects},
  author    = {Gkolemis, Vasilis and Kavouras, Loukas and Kyriakopoulos, Dimitrios
               and Tsopelas, Konstantinos and Rontogiannis, Dimitrios
               and Casalicchio, Giuseppe and Dalamagas, Theodore and Diou, Christos},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Datasets keep the licences of their sources.
