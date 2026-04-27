# Term RGNN Pipeline

This directory adds one organized entry point around the existing `RGNN/` and
`CompareTerms/` code. It keeps each experiment in its own run directory, so
datasets, models, link metrics, reconstructed terms, and term-level metrics do
not mix with older outputs.

## Quick Start

Run from the repository root:

```powershell
python .\TermRGNNPipeline\scripts\run_pipeline.py `
  --config .\TermRGNNPipeline\configs\example.yaml
```

Or run with command-line options only:

```powershell
python .\TermRGNNPipeline\scripts\run_pipeline.py `
  --data-dir data `
  --ngram 2gram `
  --run-name dataset_2gram
```

Use `--dry-run` to print every command without executing training:

```powershell
python .\TermRGNNPipeline\scripts\run_pipeline.py `
  --config .\TermRGNNPipeline\configs\example.yaml `
  --dry-run
```

The default example config currently uses:

```text
ngram: all
encoder: rgcn
decoder: complex
negative_sampling.hard_mode: mixed
negative_sampling.corrupt_mode: both
```

`hard_mode: true` in older configs is normalized to `mixed`.

To run the CompGCN encoder instead, use:

```powershell
python .\TermRGNNPipeline\scripts\run_pipeline.py `
  --config .\TermRGNNPipeline\configs\compgcn_example.yaml
```

## Input Layout

The input dataset must use the existing repository layout:

```text
data/
  Agriculture/
    train/train_term_Agriculture_2gram.txt
    dev/dev_term_Agriculture_2gram.txt
    test/test_term_Agriculture_2gram.txt
  Computer/
  Dentistry/
  Physics/
```

The same layout also works for `data_timeline/`.

## Output Layout

Each run is written under `TermRGNNPipeline/runs/{run_name}/`:

```text
runs/{run_name}/
  config.json
  logs/
  rgnn/
    datasets/{Domain}/processed_dataset.pt
    datasets/{Domain}/trained_dataset.pt
    models/{Domain}/best_model.pt
    link_metrics/{Domain}/test_metrics.csv
    postprocess/{Domain}/result/
    reconstructed_terms/{Domain}_reconstruct_terms.txt
  term_eval/
    term_metrics.csv
  summary/
    summary.csv
```

## Evaluation Outputs

The pipeline produces two evaluation layers.

1. RGNN link-level evaluation:
   - Source: `RGNN/train.py`
   - Output: `rgnn/link_metrics/{Domain}/test_metrics.csv`
   - Metrics: AUC, AP, accuracy, precision, recall, F1, optimal threshold,
     MRR, Hits@1, Hits@5, Hits@10
   - Rows: `overall`, `relation_type=1`, and `relation_type=2`

2. Term-level evaluation:
   - Source: `TermRGNNPipeline/scripts/term_eval.py`
   - Output: `term_eval/term_metrics.csv`
   - Metrics: ground-truth count, result count, true positives, precision,
     recall, F1

The final joined table is:

```text
summary/summary.csv
```

`summary.csv` includes overall link metrics and Type 2 ranking metrics:

```text
link_mrr, link_hits_at_1, link_hits_at_5, link_hits_at_10
type2_mrr, type2_hits_at_1, type2_hits_at_5, type2_hits_at_10
```

## Reconstruction Modes

Set `postprocess.reconstruction_mode` in the config.

- `predicted_term_links`: use predicted Type 2 term links from `trained_dataset.pt`.
- `prediction_only_eval`: use only predicted Type 1 composition edges and Type 2 term links.
- `full_graph`: reconstruct from the full graph edges in the selected dataset.

The default is `predicted_term_links`, which matches the goal of evaluating the
trained RGNN predictions.

Important caveat: `predicted_term_links` reconstructs terms from predicted
positive Type 2 links. Predicted negative samples are not included unless
`include_predicted_negatives: true` is set. Therefore term-level precision is
filtered by both Type 2 prediction and Type 1 path existence, and it can be
higher than the raw Type 2 false-positive behavior would suggest.

When `shortest_only: true`, reconstruction uses the shortest Type 1 path. The
current implementation does not apply `cutoff` to that shortest path. `cutoff`
only applies when all simple paths are enumerated.

## Model Settings

The training config controls the RGNN model:

```yaml
training:
  encoder: rgcn        # rgcn, rgat, or compgcn
  decoder: complex     # complex or distmult
  compgcn_composition: mult
```

`compgcn_composition` is used only when `encoder: compgcn`.

## Notes

- The term-level evaluator reads the test split directly from the input dataset.
  It does not require manually copied `CompareTerms/ground_truth_*` folders.
- If a run directory already exists, the command stops. Use `--resume` to allow
  writing into the same run directory.
- For fair RGCN vs CompGCN comparisons, keep decoder, seed, negative sampling,
  epoch count, and data split identical.
