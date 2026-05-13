# GNN を用いた用語構成語間リンク予測における non-term 利用実験レポート

## 1. 実験目的

本実験の目的は、用語データ `St` のみで GNN を学習する場合と、文書由来の non-term 候補 `Sn` を negative sample として学習に導入する場合を比較し、`Sn` の導入が link prediction 性能および term-level reconstruction 性能に与える影響を確認することである。

比較した条件は以下の 2 つである。

| 実験 | 内容 |
|---|---|
| 実験1: `exp1_st_only` | `St` の positive edge のみで学習する。`Sn` token は node vocabulary に含めるが、train/dev の negative には使わない。test negative としてのみ `Sn` を使う。 |
| 実験2: `exp2_st_sn` | `St` の positive edge と、`Sn` 由来の negative edge を train/dev/test のすべてで使う。 |

## 2. 対象データ

今回の実験対象は Agriculture のみである。`ngram: all` とし、term 側は既存の `All` ファイルを使った。

### 2.1 St: term data

| split | file |
|---|---|
| train | `data/Agriculture/train/train_term_Agriculture_All.txt` |
| dev | `data/Agriculture/dev/dev_term_Agriculture_All.txt` |
| test | `data/Agriculture/test/test_term_Agriculture_All.txt` |

### 2.2 Sn: non-term data

`nonterm_Agriculture_All.txt` は存在しないため、各 split で 1gram / 2gram / 3gram の non-term ファイルを結合して使った。

| split | files |
|---|---|
| train | `train_nonterm_Agriculture_1gram.txt`, `train_nonterm_Agriculture_2gram.txt`, `train_nonterm_Agriculture_3gram.txt` |
| dev | `dev_nonterm_Agriculture_1gram.txt`, `dev_nonterm_Agriculture_2gram.txt`, `dev_nonterm_Agriculture_3gram.txt` |
| test | `test_nonterm_Agriculture_1gram.txt`, `test_nonterm_Agriculture_2gram.txt`, `test_nonterm_Agriculture_3gram.txt` |

## 3. Edge 構築規則

edge 構築規則は既存 RGNN 実装に合わせた。

| n-gram | edge |
|---|---|
| 1-gram | Type 2 self-link |
| 2-gram 以上 | 隣接 token 間を Type 1 composition edge |
| 2-gram 以上 | 先頭 token から末尾 token を Type 2 term-link edge |

例:

```text
crop disease management
=> Type 1: crop -> disease
=> Type 1: disease -> management
=> Type 2: crop -> management
```

`St` 由来の edge は positive sample として使う。`Sn` 由来の edge candidate は positive graph edge には追加せず、negative sample としてのみ使う。また、`Sn` edge が `St` positive edge と重複した場合、その `Sn` edge は negative から除外した。

## 4. Dataset 構成

全条件で node vocabulary は `St + Sn` の token から作成した。vocabulary size は 6041 である。

| split | positive St edges | 実験1 negative | 実験2 negative |
|---|---:|---:|---:|
| train | 6363 | 6363 random negatives | 5360 Sn negatives |
| dev | 897 | 897 random negatives | 646 Sn negatives |
| test | 1788 | 1269 Sn negatives | 1269 Sn negatives |

`Sn` edge は positive edge との重複を除去した。positive と重複していた `Sn` edge 数は train 322、dev 46、test 92 であった。

実験1と実験2では、test positive と test negative を同じにしている。したがって、test 時の比較条件は揃っている。

## 5. 学習設定

共通設定は以下の通りである。

| 項目 | 値 |
|---|---|
| domain | Agriculture |
| ngram | all |
| encoder | RGCN |
| decoder | ComplEx |
| hidden dim | 64 |
| embedding dim | 32 |
| learning rate | 0.01 |
| seed | 42 |
| eval every | 10 epochs |
| negative ratio | 1 |
| explicit Sn negative policy | `match_positive` |
| early stopping | 無効、`early_stopping_patience: 0` |
| final evaluation | validation AUC が最良の checkpoint をロードして test 評価 |

node feature 条件として以下を比較した。

| 条件 | node feature |
|---|---|
| fastText なし | identity feature |
| fastText あり | `data/cc.en.300.bin` 由来の 300 次元 fastText vector |

epoch 条件として 100 epoch と 1000 epoch を実行した。

## 6. 実行した run

| 条件 | run directory |
|---|---|
| fastText なし, epoch 100 | `experiments/nonterm_link_prediction/runs/agriculture_nonterm_all` |
| fastText なし, epoch 1000 | `experiments/nonterm_link_prediction/runs/agriculture_nonterm_all_epoch1000` |
| fastText あり, epoch 100 | `experiments/nonterm_link_prediction/runs/agriculture_nonterm_all_fasttext_epoch100` |
| fastText あり, epoch 1000 | `experiments/nonterm_link_prediction/runs/agriculture_nonterm_all_fasttext_epoch1000` |

## 7. Link-Level 評価結果

### 7.1 fastText なし

| epoch | experiment | AUC | AP | Accuracy | Precision | Recall | F1 | best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | exp1_st_only | 0.6560 | 0.7362 | 0.5924 | 0.6071 | 0.8591 | 0.7114 | 20 |
| 100 | exp2_st_sn | 0.8513 | 0.8700 | 0.7347 | 0.8438 | 0.6706 | 0.7473 | 20 |
| 1000 | exp1_st_only | 0.6561 | 0.7362 | 0.5924 | 0.6071 | 0.8591 | 0.7114 | 20 |
| 1000 | exp2_st_sn | 0.8513 | 0.8700 | 0.7347 | 0.8438 | 0.6706 | 0.7473 | 20 |

fastText なしでは、実験2が実験1より link-level AUC、AP、Accuracy、Precision、F1 で高い。Recall は実験1の方が高いが、実験1はより多くの edge を positive と判定する傾向があり、Precision が低い。

100 epoch と 1000 epoch で結果がほぼ同じなのは、どちらの run でも best validation checkpoint が epoch 20 だったためである。

### 7.2 fastText あり

| epoch | experiment | AUC | AP | Accuracy | Precision | Recall | F1 | best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | exp1_st_only | 0.6997 | 0.7574 | 0.6428 | 0.7644 | 0.5626 | 0.6482 | 90 |
| 100 | exp2_st_sn | 0.8566 | 0.8834 | 0.7311 | 0.8827 | 0.6230 | 0.7305 | 80 |
| 1000 | exp1_st_only | 0.6899 | 0.7449 | 0.6264 | 0.7609 | 0.5268 | 0.6226 | 210 |
| 1000 | exp2_st_sn | 0.8623 | 0.8863 | 0.7360 | 0.8853 | 0.6303 | 0.7364 | 240 |

fastText ありでも、実験2は一貫して実験1を上回った。特に epoch 1000 の実験2が link-level では最良で、AUC 0.8623、AP 0.8863、F1 0.7364 であった。

## 8. Term-Level 評価結果

### 8.1 fastText なし

| epoch | experiment | GT terms | reconstructed | true positives | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | exp1_st_only | 1178 | 1005 | 984 | 0.979104 | 0.835314 | 0.901512 |
| 100 | exp2_st_sn | 1178 | 642 | 617 | 0.961059 | 0.523769 | 0.678022 |
| 1000 | exp1_st_only | 1178 | 1005 | 984 | 0.979104 | 0.835314 | 0.901512 |
| 1000 | exp2_st_sn | 1178 | 642 | 617 | 0.961059 | 0.523769 | 0.678022 |

fastText なしでは、term-level では実験1が大きく高い。実験2は link-level の negative 識別性能は上がったが、reconstructed term 数が 1005 から 642 に減少し、Recall が大きく下がった。

### 8.2 fastText あり

| epoch | experiment | GT terms | reconstructed | true positives | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | exp1_st_only | 1178 | 592 | 572 | 0.966216 | 0.485569 | 0.646328 |
| 100 | exp2_st_sn | 1178 | 592 | 572 | 0.966216 | 0.485569 | 0.646328 |
| 1000 | exp1_st_only | 1178 | 555 | 538 | 0.969369 | 0.456706 | 0.620889 |
| 1000 | exp2_st_sn | 1178 | 605 | 587 | 0.970248 | 0.498302 | 0.658441 |

fastText ありでは、epoch 100 では実験1と実験2の term-level 結果が同一であった。epoch 1000 では実験2が実験1より少し良く、Term F1 は 0.620889 から 0.658441 に上がった。

ただし、fastText ありの term-level F1 は fastText なしの実験1より低い。fastText により link-level AUC は改善する傾向がある一方で、term reconstruction に必要な Type 1 path と Type 2 target pair の組み合わせが十分に復元されず、reconstructed term 数が少なくなった可能性がある。

## 9. 実験2の改善量

### 9.1 Link-Level F1

| 条件 | exp1 F1 | exp2 F1 | exp2 - exp1 |
|---|---:|---:|---:|
| fastText なし, epoch 100 | 0.7114 | 0.7473 | +0.0359 |
| fastText なし, epoch 1000 | 0.7114 | 0.7473 | +0.0359 |
| fastText あり, epoch 100 | 0.6482 | 0.7305 | +0.0823 |
| fastText あり, epoch 1000 | 0.6226 | 0.7364 | +0.1138 |

### 9.2 Term-Level F1

| 条件 | exp1 F1 | exp2 F1 | exp2 - exp1 |
|---|---:|---:|---:|
| fastText なし, epoch 100 | 0.901512 | 0.678022 | -0.223490 |
| fastText なし, epoch 1000 | 0.901512 | 0.678022 | -0.223490 |
| fastText あり, epoch 100 | 0.646328 | 0.646328 | 0.000000 |
| fastText あり, epoch 1000 | 0.620889 | 0.658441 | +0.037552 |

## 10. 分析

### 10.1 Sn の training 導入は link prediction には有効

全条件で実験2は実験1より link-level AUC、AP、Precision、F1 が高い。これは、`Sn` を negative edge として明示的に学習に入れることで、GNN が term を構成する語の関係と non-term 候補の語関係をより区別できるようになったことを示している。

特に fastText あり epoch 1000 では、実験2が AUC 0.8623、AP 0.8863、F1 0.7364 を記録しており、今回の link-level 評価では最良であった。

### 10.2 link-level 改善は term-level 改善に直結しない

term-level evaluation は、単に Type 2 edge を正しく分類できるかだけでは決まらない。既存の reconstruction は Type 1 composition graph 上の path と Type 2 term-link edge の組み合わせに依存する。

そのため、link-level で negative 識別が改善しても、復元対象となる Type 2 link や Type 1 path が十分に残らなければ reconstructed term 数が減り、term recall が下がる。

fastText なしの実験2では、この影響が顕著である。実験1では 1005 terms を復元し 984 true positives を得たが、実験2では 642 terms の復元に留まり、true positives も 617 まで下がった。

### 10.3 fastText は link-level には概ね有効だが、term-level では慎重な解釈が必要

fastText ありでは、実験2の link-level AUC/AP は fastText なしよりわずかに高い。一方で、term-level では fastText なしの実験1が最も高い F1 を示した。

これは、fastText による semantic feature が edge scoring には有効でも、現在の reconstruction pipeline では predicted edge の閾値選択、Type 1 connectivity、Type 2 target selection の影響を強く受けるためである。

### 10.4 epoch 1000 の効果

fastText なしでは、epoch 100 と epoch 1000 の最終結果はほぼ同一であった。best checkpoint がどちらも epoch 20 だったためである。

fastText ありでは、epoch 1000 にすることで実験2の link-level AUC/AP/F1 は少し改善した。best checkpoint は exp1 が epoch 210、exp2 が epoch 240 であり、fastText ありでは 100 epoch より長い学習が validation AUC 上は有効だった。

ただし、epoch 1000 でも最終 epoch のモデルではなく best validation checkpoint で評価している点に注意が必要である。

## 11. 結論

今回の Agriculture / all / RGCN + ComplEx 条件では、`Sn` を negative sample として学習に導入することは、link-level edge prediction 性能の向上に一貫して有効であった。

一方で、term-level reconstruction では必ずしも同じ改善が得られなかった。特に fastText なしでは、実験2は link-level では改善したが、reconstructed term 数と term recall が大きく低下した。

したがって、現時点での結論は以下である。

1. `Sn` の導入は link prediction の識別性能を改善する。
2. term-level 性能を改善するには、link classifier だけでなく reconstruction 側の設計も調整する必要がある。
3. fastText は link-level では有効な可能性が高いが、term-level では閾値・Type 1 path・Type 2 target selection との相互作用を確認する必要がある。

## 12. 次に確認すべき点

今後の確認項目として、以下が重要である。

| 項目 | 目的 |
|---|---|
| Type 1 / Type 2 別の Precision / Recall / F1 分析 | term reconstruction 低下が composition edge 由来か term-link edge 由来かを切り分ける |
| fixed threshold 0.5 での評価 | validation F1 最適閾値が低すぎる影響を確認する |
| predicted Type 2 数と reconstructed term 数の差分分析 | Type 2 は予測されているが Type 1 path がなく復元できないケースを数える |
| `include_predicted_negatives` の影響確認 | false positive negative edge が term-level にどう影響するか確認する |
| Sn negative ratio の感度分析 | `1:1`、`1:5`、`all` の比較 |
| 他ドメインへの拡張 | Agriculture の傾向が Computer Science / Dentistry / Physics でも再現するか確認する |

