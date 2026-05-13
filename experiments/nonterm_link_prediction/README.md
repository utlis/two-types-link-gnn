# Non-Term Link Prediction Experiment

This experiment compares two RGNN link-prediction settings:

1. `exp1_st_only`: train with terminology positive edges from `St` only. Non-term
   `Sn` tokens are included in the node vocabulary, but `Sn` edges are used only
   as test negatives.
2. `exp2_st_sn`: train with terminology positive edges from `St` and explicit
   non-term negative edges from `Sn`. The test split is the same as
   `exp1_st_only`.

The RGNN architecture, trainer, reconstruction, and term-level evaluator are the
existing repository modules. This directory only builds experiment-specific
datasets and runs those modules.

## Edge Construction Rule

The scripts follow the current RGNN rule:

- 1-gram: one Type 2 self-link.
- 2+-gram: adjacent token pairs become Type 1 composition edges.
- 2+-gram: the first token to last token becomes one Type 2 term-link edge.

`St` edges are positive samples. `Sn` edges are negative samples and are not
added to the positive message-passing graph.

## Quick Start

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\experiments\nonterm_link_prediction\scripts\run_nonterm_experiments.py `
  --config .\experiments\nonterm_link_prediction\configs\agriculture_2gram.yaml
```

To inspect commands without running training:

```powershell
.\.venv\Scripts\python.exe .\experiments\nonterm_link_prediction\scripts\run_nonterm_experiments.py `
  --config .\experiments\nonterm_link_prediction\configs\agriculture_2gram.yaml `
  --dry-run
```

Outputs are written to:

```text
experiments/nonterm_link_prediction/runs/{run_name}/
  exp1_st_only/
  exp2_st_sn/
  comparison/
```

Each experiment directory contains RGNN datasets, checkpoints, link metrics,
reconstructed terms, term-level metrics, and a summary CSV.

## Input Layout

The default config uses Agriculture 2-grams:

```text
data/Agriculture/train/train_term_Agriculture_2gram.txt
data/Agriculture/dev/dev_term_Agriculture_2gram.txt
data/Agriculture/test/test_term_Agriculture_2gram.txt
data/Agriculture/train/train_nonterm_Agriculture_2gram.txt
data/Agriculture/dev/dev_nonterm_Agriculture_2gram.txt
data/Agriculture/test/test_nonterm_Agriculture_2gram.txt
```

For `ngram: all`, term `All` files are used when present. If
`nonterm_*_All.txt` does not exist, the script merges the available 1/2/3-gram
non-term files for that split.

## Notes

- The initial negative ratio is `1:1` by default. Set
  `negative_sampling.explicit_policy: all` to use all available `Sn` edges.
- `Sn` edge candidates that overlap any `St` positive edge are removed from the
  negative sets.
- For `exp1_st_only`, train/dev negatives are random negatives, but any edge
  candidate that appears in `Sn` is forbidden. Only test negatives come from
  `Sn`.
- For fair comparison, both experiments use the same `St` test positives and the
  same sampled `Sn` test negatives.
- FastText node-feature matrices are cached by vocabulary and embedding-file
  metadata under `experiments/nonterm_link_prediction/.cache/embeddings/`.
