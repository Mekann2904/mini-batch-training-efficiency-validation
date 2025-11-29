<!-- /README.md -->
<!-- プロジェクト概要と実行手順をまとめる。 -->
<!-- ベンチマークの使い方を最短で共有するために存在する。 -->
<!-- 関連ファイル: 定義書.md, plans.md, benchmark_batch.py -->

# mini-batch-training-efficiency-validation

PythonとNumPyでフルバッチとミニバッチの処理時間を比較するシンプルなベンチマークです。


## セットアップ

- Python 3.x
- NumPy
- scikit-learn（`fetch_openml('mnist_784')` で自動取得）

scikit-learnが無い場合や取得に失敗した場合、書籍付属の `dataset.mnist`、さらに無い場合は同形状のダミーデータへ順にフォールバックします（計測値は目安）。


## 実行

```bash
python benchmark_batch.py
```

- 5回計測の平均を表示します。
- それぞれの5回分の計測値も表示します。
- `benchmark_plot.png` に平均±標準偏差の棒グラフと各試行のドットを保存します（対数スケール、matplotlib必要）。


## 使い方

```bash
python benchmark_batch.py
```

コンソールにフルバッチとミニバッチの計測結果とスピードアップ倍率が表示されます。
