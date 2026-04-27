# Term RGNN Project

このリポジトリは、用語データセットから Type 1 / Type 2 edge を持つグラフを作成し、RGNN / CompGCN による link prediction と、復元された term-level evaluation をまとめて実行するためのプロジェクトです。

## Overall Workflow

```mermaid
flowchart TD
    A["Input dataset<br/>data/ or data_timeline/"] --> B["Dataset creation<br/>RGNN/create_dataset.py"]
    B --> C["Graph dataset<br/>processed_dataset.pt"]

    C --> D["GNN training<br/>RGNN/train.py"]
    D --> E["Best checkpoint<br/>best_model.pt"]
    D --> F["Link prediction metrics<br/>test_metrics.csv"]
    D --> G["Predicted graph output<br/>trained_dataset.pt"]

    G --> H["Term reconstruction<br/>RGNN/reconstruct_terms.py"]
    H --> I["Reconstructed terms<br/>*_reconstruct_terms.txt"]

    I --> J["Term-level evaluation<br/>TermRGNNPipeline/scripts/term_eval.py"]
    F --> K["Summary join<br/>TermRGNNPipeline/scripts/summarize.py"]
    J --> K

    K --> L["Final outputs<br/>summary.csv"]
```

## Edge Types

```mermaid
flowchart LR
    S["start token"] -- "Type 1 edge<br/>composition path" --> M["middle token(s)"]
    M -- "Type 1 edge<br/>composition path" --> T["end token"]
    S -. "Type 2 edge<br/>term target pair" .-> T
```

| Project label | Internal id | Role |
| --- | ---: | --- |
| Type 1 | 0 | 単語列をどうつなぐかを表す composition edge |
| Type 2 | 1 | 始点から終点までを term として復元したい対象ペア |

## Main Directories

```mermaid
flowchart TD
    R["Repository root"] --> DATA["data/<br/>data_timeline/"]
    R --> PIPE["TermRGNNPipeline/"]
    R --> RGNN["RGNN/"]
    R --> COMP["CompareTerms/"]

    DATA --> D1["Domain splits<br/>train / dev / test"]

    PIPE --> P1["configs/<br/>experiment settings"]
    PIPE --> P2["scripts/run_pipeline.py<br/>single entry point"]
    PIPE --> P3["runs/<br/>generated outputs"]

    RGNN --> R1["data/<br/>graph builder and dataset"]
    RGNN --> R2["models/<br/>RGCN / RGAT / CompGCN encoders<br/>ComplEx / DistMult decoders"]
    RGNN --> R3["train.py<br/>link prediction training"]
    RGNN --> R4["reconstruct_terms.py<br/>term reconstruction"]

    COMP --> C1["legacy comparison scripts"]
```

## File Relationship

```mermaid
flowchart TD
    CFG["TermRGNNPipeline/configs/example.yaml"] --> RUN["TermRGNNPipeline/scripts/run_pipeline.py"]
    RUN --> STAGE["TermRGNNPipeline/scripts/stages.py"]

    STAGE --> CREATE["RGNN/create_dataset.py"]
    CREATE --> BUILDER["RGNN/data/graph_builder.py"]
    CREATE --> DATASET["RGNN/data/dataset.py"]
    BUILDER --> PT["processed_dataset.pt"]
    DATASET --> PT

    STAGE --> TRAIN["RGNN/train.py"]
    TRAIN --> MODEL["RGNN/models/rgcn_distmult.py"]
    MODEL --> RGCN["RGNN/models/rgcn_encoder.py"]
    MODEL --> RGAT["RGNN/models/rgat_encoder.py"]
    MODEL --> COMPGCN["RGNN/models/compgcn_encoder.py"]
    TRAIN --> LINK["test_metrics.csv"]
    TRAIN --> TRAINED["trained_dataset.pt"]

    STAGE --> RECON["RGNN/reconstruct_terms.py"]
    TRAINED --> RECON
    RECON --> TERMS["reconstructed_terms.txt"]

    STAGE --> TERMEVAL["TermRGNNPipeline/scripts/term_eval.py"]
    TERMS --> TERMEVAL
    TERMEVAL --> TMETRICS["term_metrics.csv"]

    STAGE --> SUMMARY["TermRGNNPipeline/scripts/summarize.py"]
    LINK --> SUMMARY
    TMETRICS --> SUMMARY
    SUMMARY --> FINAL["summary.csv"]
```

## How To Run

ルートディレクトリから実行します。

```powershell
.\.venv\Scripts\python.exe .\TermRGNNPipeline\scripts\run_pipeline.py `
  --config .\TermRGNNPipeline\configs\example.yaml
```

CompGCN encoder を使う場合は、次の config を使います。

```powershell
.\.venv\Scripts\python.exe .\TermRGNNPipeline\scripts\run_pipeline.py `
  --config .\TermRGNNPipeline\configs\compgcn_example.yaml
```

## Evaluation Outputs

```mermaid
flowchart LR
    A["Link prediction evaluation"] --> A1["AUC / AP / Accuracy"]
    A --> A2["Precision / Recall / F1"]
    A --> A3["MRR / Hits@1 / Hits@5 / Hits@10"]

    B["Term-level evaluation"] --> B1["Term precision"]
    B --> B2["Term recall"]
    B --> B3["Term F1"]

    A1 --> C["summary.csv"]
    A2 --> C
    A3 --> C
    B1 --> C
    B2 --> C
    B3 --> C
```

最終的な評価結果は `TermRGNNPipeline/runs/{run_name}/summary/summary.csv` にまとめられます。

## References

- `TermRGNNPipeline/README.md`: pipeline の実行方法、config、出力構造
- `RGNN/README.md`: GNN 内部処理、edge type、negative sampling、ranking evaluation
- `.gitignore`: commit から除外する中間生成物、学習済みモデル、キャッシュ類
