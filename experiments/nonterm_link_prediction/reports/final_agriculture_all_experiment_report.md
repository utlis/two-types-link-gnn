# 最終比較レポート: Agriculture all における non-term negative 利用実験

## 1. 目的

本実験の目的は、GNN による用語構成語間 link prediction において、文書由来の non-term 候補 `Sn` を negative sample として学習に導入することが有効かを検証することである。

比較対象は以下の 2 条件である。

| 実験 | 内容 |
|---|---|
| 実験1: `exp1_st_only` | `St` 由来の positive edge で学習する。train/dev negative は random negative。ただし、修正後実装では random negative にも `Sn` edge candidate が混入しないように除外した。test negative は `Sn` 由来。 |
| 実験2: `exp2_st_sn` | `St` 由来の positive edge と、`Sn` 由来の negative edge を train/dev/test で使う。 |

今回の結果は、レビュー後に修正した以下の条件で再実行した最終比較である。

1. 実験1の random negative から `Sn` edge candidate を除外。
2. `train.py` に渡す学習ハイパーパラメータを run config に明示。
3. fastText embedding matrix を vocabulary ごとに cache。

## 2. 対象データ

対象 domain は Agriculture のみである。`ngram: all` とし、term 側は既存の `All` ファイルを使用した。

### 2.1 St: term data

| split | file |
|---|---|
| train | `data/Agriculture/train/train_term_Agriculture_All.txt` |
| dev | `data/Agriculture/dev/dev_term_Agriculture_All.txt` |
| test | `data/Agriculture/test/test_term_Agriculture_All.txt` |

### 2.2 Sn: non-term data

`nonterm_Agriculture_All.txt` は存在しないため、各 split で 1gram / 2gram / 3gram の non-term ファイルを結合した。

| split | files |
|---|---|
| train | `train_nonterm_Agriculture_1gram.txt`, `train_nonterm_Agriculture_2gram.txt`, `train_nonterm_Agriculture_3gram.txt` |
| dev | `dev_nonterm_Agriculture_1gram.txt`, `dev_nonterm_Agriculture_2gram.txt`, `dev_nonterm_Agriculture_3gram.txt` |
| test | `test_nonterm_Agriculture_1gram.txt`, `test_nonterm_Agriculture_2gram.txt`, `test_nonterm_Agriculture_3gram.txt` |

## 3. Edge 作成方法

edge 作成規則は既存 RGNN 実装に合わせた。

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

`St` 由来 edge は positive sample として使う。`Sn` 由来 edge candidate は positive graph edge には追加せず、negative sample として使う。`Sn` edge candidate が `St` positive edge と重複する場合は、negative から除外した。

## 4. Dataset 構成

node vocabulary は全条件で `St + Sn` の token から作成した。vocabulary size は 6041 である。

| split | positive St edges | 実験1 negative | 実験2 negative |
|---|---:|---:|---:|
| train | 6363 | 6363 random negatives, excluding all Sn candidates | 5360 Sn negatives |
| dev | 897 | 897 random negatives, excluding all Sn candidates | 646 Sn negatives |
| test | 1788 | 1269 Sn negatives | 1269 Sn negatives |

test edge の Type 別内訳は以下である。

| edge set | Type 1 | Type 2 | total |
|---|---:|---:|---:|
| test positive, St 由来 | 897 | 891 | 1788 |
| test negative, Sn 由来 | 714 | 555 | 1269 |

実験1と実験2では test positive / test negative を同一にしている。したがって、test 条件は揃っている。

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
| dropout | 0.2 |
| learning rate | 0.01 |
| loss | balanced BCE |
| weight decay | 0.0001 |
| seed | 42 |
| eval every | 10 epochs |
| early stopping | disabled, `early_stopping_patience: 0` |
| final evaluation | validation AUC が最良の checkpoint をロードして test 評価 |

node feature は以下の 2 条件で比較した。

| 条件 | node feature |
|---|---|
| fastText なし | identity feature |
| fastText あり | `data/cc.en.300.bin` による 300 次元 fastText vector |

epoch は 100 と 1000 を実行した。

## 6. 実行 run

| 条件 | run directory |
|---|---|
| fastText なし, epoch 100 | `experiments/nonterm_link_prediction/runs/final_agriculture_nonterm_all_epoch100` |
| fastText なし, epoch 1000 | `experiments/nonterm_link_prediction/runs/final_agriculture_nonterm_all_epoch1000` |
| fastText あり, epoch 100 | `experiments/nonterm_link_prediction/runs/final_agriculture_nonterm_all_fasttext_epoch100` |
| fastText あり, epoch 1000 | `experiments/nonterm_link_prediction/runs/final_agriculture_nonterm_all_fasttext_epoch1000` |

集計 CSV は以下に出力した。

| 内容 | file |
|---|---|
| overall 指標 | `experiments/nonterm_link_prediction/reports/final_overall_results.csv` |
| Type 1 / Type 2 別指標 | `experiments/nonterm_link_prediction/reports/final_type_metrics.csv` |
| predicted Type 2 と reconstructed term 数 | `experiments/nonterm_link_prediction/reports/final_reconstruction_gap.csv` |

## 7. Overall Link-Level 結果

| feature | epoch | experiment | best epoch | AUC | AP | Accuracy | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| no fastText | 100 | exp1_st_only | 100 | 0.6359 | 0.7323 | 0.5777 | 0.6062 | 0.7936 | 0.6873 |
| no fastText | 100 | exp2_st_sn | 20 | 0.8513 | 0.8700 | 0.7347 | 0.8438 | 0.6706 | 0.7473 |
| no fastText | 1000 | exp1_st_only | 640 | 0.5823 | 0.6876 | 0.5819 | 0.7083 | 0.4849 | 0.5757 |
| no fastText | 1000 | exp2_st_sn | 20 | 0.8513 | 0.8700 | 0.7347 | 0.8438 | 0.6706 | 0.7473 |
| fastText | 100 | exp1_st_only | 90 | 0.6859 | 0.7495 | 0.6254 | 0.7498 | 0.5397 | 0.6276 |
| fastText | 100 | exp2_st_sn | 80 | 0.8566 | 0.8834 | 0.7311 | 0.8827 | 0.6230 | 0.7305 |
| fastText | 1000 | exp1_st_only | 910 | 0.6816 | 0.7410 | 0.6238 | 0.7552 | 0.5280 | 0.6215 |
| fastText | 1000 | exp2_st_sn | 280 | 0.8578 | 0.8835 | 0.7363 | 0.8812 | 0.6348 | 0.7380 |

## 8. Overall Term-Level 結果

| feature | epoch | experiment | GT terms | reconstructed | true positives | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| no fastText | 100 | exp1_st_only | 1178 | 941 | 916 | 0.973433 | 0.777589 | 0.864559 |
| no fastText | 100 | exp2_st_sn | 1178 | 642 | 617 | 0.961059 | 0.523769 | 0.678022 |
| no fastText | 1000 | exp1_st_only | 1178 | 485 | 457 | 0.942268 | 0.387946 | 0.549609 |
| no fastText | 1000 | exp2_st_sn | 1178 | 642 | 617 | 0.961059 | 0.523769 | 0.678022 |
| fastText | 100 | exp1_st_only | 1178 | 544 | 520 | 0.955882 | 0.441426 | 0.603949 |
| fastText | 100 | exp2_st_sn | 1178 | 592 | 572 | 0.966216 | 0.485569 | 0.646328 |
| fastText | 1000 | exp1_st_only | 1178 | 582 | 570 | 0.979381 | 0.483871 | 0.647727 |
| fastText | 1000 | exp2_st_sn | 1178 | 605 | 587 | 0.970248 | 0.498302 | 0.658441 |

## 9. Type 1 / Type 2 別の P, R, F1 分析

### 9.1 fastText なし

| epoch | experiment | type | Precision | Recall | F1 | AUC | AP |
|---:|---|---|---:|---:|---:|---:|---:|
| 100 | exp1_st_only | Type 1 | 0.5488 | 0.6961 | 0.6137 | 0.6315 | 0.7163 |
| 100 | exp1_st_only | Type 2 | 0.6353 | 0.8456 | 0.7255 | 0.6250 | 0.7380 |
| 100 | exp2_st_sn | Type 1 | 0.8491 | 0.8778 | 0.8632 | 0.9165 | 0.9190 |
| 100 | exp2_st_sn | Type 2 | 0.8393 | 0.5600 | 0.6718 | 0.8226 | 0.8509 |
| 1000 | exp1_st_only | Type 1 | 0.6919 | 0.5740 | 0.6274 | 0.5967 | 0.6815 |
| 1000 | exp1_st_only | Type 2 | 0.7203 | 0.4374 | 0.5443 | 0.5798 | 0.6955 |
| 1000 | exp2_st_sn | Type 1 | 0.8491 | 0.8778 | 0.8632 | 0.9165 | 0.9190 |
| 1000 | exp2_st_sn | Type 2 | 0.8393 | 0.5600 | 0.6718 | 0.8226 | 0.8509 |

fastText なしでは、実験2の Type 1 性能が非常に大きく改善している。Type 1 F1 は epoch 100 で 0.6137 から 0.8632 に上がった。これは `Sn` 由来の隣接 token pair を negative として学習したことで、composition edge の識別が強く改善したことを示す。

一方、Type 2 は解釈がやや異なる。epoch 100 では、実験1の Type 2 recall が 0.8456 と高く、実験2では 0.5600 に下がる。しかし precision は 0.6353 から 0.8393 に大きく上がる。つまり、実験2は Type 2 term-link をより保守的に予測し、false positive を減らすが、positive Type 2 の取りこぼしも増やす傾向がある。

### 9.2 fastText あり

| epoch | experiment | type | Precision | Recall | F1 | AUC | AP |
|---:|---|---|---:|---:|---:|---:|---:|
| 100 | exp1_st_only | Type 1 | 0.7559 | 0.6222 | 0.6825 | 0.7117 | 0.7686 |
| 100 | exp1_st_only | Type 2 | 0.7458 | 0.4957 | 0.5956 | 0.6653 | 0.7436 |
| 100 | exp2_st_sn | Type 1 | 0.8696 | 0.8151 | 0.8415 | 0.9138 | 0.9156 |
| 100 | exp2_st_sn | Type 2 | 0.8940 | 0.5206 | 0.6580 | 0.8374 | 0.8750 |
| 1000 | exp1_st_only | Type 1 | 0.7484 | 0.5595 | 0.6403 | 0.6825 | 0.7325 |
| 1000 | exp1_st_only | Type 2 | 0.7592 | 0.5111 | 0.6110 | 0.6753 | 0.7463 |
| 1000 | exp2_st_sn | Type 1 | 0.8763 | 0.8312 | 0.8531 | 0.9103 | 0.9127 |
| 1000 | exp2_st_sn | Type 2 | 0.8854 | 0.5300 | 0.6631 | 0.8356 | 0.8732 |

fastText ありでも、実験2は Type 1 / Type 2 の両方で F1 が上がっている。特に Type 1 は 0.84 から 0.85 程度で安定して高い。Type 2 は precision が高く、recall が低めという傾向が残る。

この結果から、`Sn` training は Type 1 composition edge の識別に特に強く効いている。一方、term reconstruction に直結する Type 2 term-link は、precision は改善するが recall が抑えられやすい。

## 10. Predicted Type 2 数と reconstructed term 数の差分分析

ここでの predicted Type 2 数は、`trained_dataset.pt` の `predicted_test_edges` のうち Type 2 の数である。現在の reconstruction mode は `predicted_term_links` であり、`include_predicted_negatives: false` のため、false positive negative edge は term reconstruction の target には使っていない。

| feature | epoch | experiment | predicted Type 1 | predicted Type 2 | false positive negative Type 1 | false positive negative Type 2 | reconstructed terms | predicted Type 2 - reconstructed |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| no fastText | 100 | exp1_st_only | 433 | 986 | 356 | 566 | 941 | 45 |
| no fastText | 100 | exp2_st_sn | 546 | 653 | 97 | 125 | 642 | 11 |
| no fastText | 1000 | exp1_st_only | 357 | 510 | 159 | 198 | 485 | 25 |
| no fastText | 1000 | exp2_st_sn | 546 | 653 | 97 | 125 | 642 | 11 |
| fastText | 100 | exp1_st_only | 387 | 578 | 125 | 197 | 544 | 34 |
| fastText | 100 | exp2_st_sn | 507 | 607 | 76 | 72 | 592 | 15 |
| fastText | 1000 | exp1_st_only | 348 | 596 | 117 | 189 | 582 | 14 |
| fastText | 1000 | exp2_st_sn | 517 | 618 | 73 | 80 | 605 | 13 |

predicted Type 2 と reconstructed term 数の差は比較的小さい。多くの条件で差は 11 から 45 程度であり、predicted Type 2 target の大半は reconstruction に到達している。

したがって、term-level recall 低下の主因は、「predicted Type 2 は多いが Type 1 path がなく復元できない」ことではない。むしろ、predicted Type 2 数そのものが少ないことが大きい。例えば no fastText epoch 100 では、実験1は predicted Type 2 が 986 あるのに対し、実験2は 653 である。この差が reconstructed term 数 941 対 642、term recall 0.7776 対 0.5238 の差にほぼ対応している。

また、false positive negative Type 2 は実験2で大きく減っている。no fastText epoch 100 では 566 から 125、fastText epoch 100 では 197 から 72 に減少している。これは `Sn` training によって non-term Type 2 edge を positive と誤判定しにくくなったことを示す。

## 11. 実験2の改善量

| 条件 | exp1 Link F1 | exp2 Link F1 | delta | exp1 Term F1 | exp2 Term F1 | delta |
|---|---:|---:|---:|---:|---:|---:|
| no fastText, epoch 100 | 0.6873 | 0.7473 | +0.0600 | 0.864559 | 0.678022 | -0.186537 |
| no fastText, epoch 1000 | 0.5757 | 0.7473 | +0.1716 | 0.549609 | 0.678022 | +0.128413 |
| fastText, epoch 100 | 0.6276 | 0.7305 | +0.1029 | 0.603949 | 0.646328 | +0.042379 |
| fastText, epoch 1000 | 0.6215 | 0.7380 | +0.1165 | 0.647727 | 0.658441 | +0.010714 |

実験2は全条件で Link F1 を改善した。Term F1 については、no fastText epoch 100 では実験1が高いが、それ以外の 3 条件では実験2が実験1を上回った。

## 12. 分析

### 12.1 `Sn` training は link-level で一貫して有効

最終比較では、全条件で実験2が overall Link F1 を改善した。AUC / AP でも同様に実験2が高い。

特に Type 1 では改善が明確である。`Sn` の n-gram から作られる隣接 token pair を negative として学習したことで、GNN は「用語の構成語として隣接する語関係」と「non-term 候補内で隣接しているだけの語関係」をより区別できるようになったと考えられる。

### 12.2 Type 2 は precision と recall の trade-off が強い

実験2では Type 2 precision が高くなる一方、recall は低めに出る傾向がある。これは `Sn` 由来 negative を学習に入れたことで、term-link edge を positive と判定する基準が厳しくなったためと考えられる。

この傾向は term reconstruction に直接影響する。reconstruction は predicted Type 2 を target とするため、Type 2 recall が下がると復元候補自体が減る。

### 12.3 term-level 性能は predicted Type 2 数に強く依存する

predicted Type 2 数と reconstructed term 数の差は小さい。したがって、復元失敗の多くは Type 1 path 欠落よりも、そもそも Type 2 が predicted positive にならないことに起因している可能性が高い。

実験2は false positive negative Type 2 を大きく減らすため、link-level precision は高くなる。一方で、positive Type 2 の predicted 数も抑えられるため、term recall が低下する場合がある。

### 12.4 fastText の効果

fastText ありでは、実験2の link-level は安定して高く、epoch 1000 で Link F1 0.7380 を得た。fastText なしの実験2と比べると、Link AUC/AP はわずかに高いが、Link F1 は同程度である。

term-level では、fastText ありの実験2が epoch 100 / 1000 の両方で実験1を上回った。ただし、最も高い Term F1 は no fastText epoch 100 の実験1であり、fastText が term reconstruction に常に有利とは言えない。

### 12.5 epoch 1000 の効果

実験2では no fastText の best epoch は 20 のままで、epoch 100 と 1000 の結果は同じであった。fastText ありでは best epoch が 80 から 280 へ伸び、epoch 1000 の方がわずかに Link F1 / Term F1 とも改善した。

実験1では epoch 1000 で best epoch が後ろにずれたが、no fastText では性能が下がった。random negative のみで長く学習すると、validation AUC 上の best checkpoint が後半に出ても、test 上では generalization が改善しない可能性がある。

## 13. 結論

Agriculture / all / RGCN + ComplEx 条件において、`Sn` を negative sample として training に導入することは、link-level edge prediction 性能の改善に一貫して有効であった。

Type 別に見ると、効果は特に Type 1 composition edge で大きい。Type 2 term-link edge でも precision は改善するが、recall が下がりやすく、term reconstruction の recall 低下につながる場合がある。

predicted Type 2 数と reconstructed term 数の差分を見ると、reconstruction 失敗そのものよりも、predicted Type 2 数の増減が term-level recall を左右している。したがって、今後は Type 2 の閾値調整や Type 1 / Type 2 別 threshold の導入が重要である。

最終的な結論は以下である。

1. `Sn` training は link-level では明確に有効。
2. Type 1 edge では特に大きな改善がある。
3. Type 2 edge は precision が上がる一方で recall が下がりやすい。
4. term-level 改善には、Type 2 predicted positive 数を適切に保つ設計が必要。
5. fastText は実験2では安定した改善を示すが、term reconstruction では閾値や復元手順との相互作用が大きい。

## 14. 今後の確認項目

| 項目 | 目的 |
|---|---|
| Type 1 / Type 2 別 threshold | Type 2 recall を保ちながら Type 1 precision を高める |
| fixed threshold 0.5 との比較 | validation F1 最適 threshold が低すぎる影響を調べる |
| Type 2 専用の validation criterion | overall AUC ではなく term reconstruction に近い criterion で checkpoint を選ぶ |
| predicted Type 2 の false negative 分析 | どの term-link が落ちているかを確認する |
| Sn negative ratio の感度分析 | `match_positive`, `all`, `1:5` などを比較する |
| 他 domain への拡張 | Agriculture の傾向が Computer Science / Dentistry / Physics でも再現するか確認する |

