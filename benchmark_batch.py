# /benchmark_batch.py
# MNISTのフルバッチとミニバッチの計測スクリプト。
# ミニバッチ学習の速度優位を簡易に可視化するために存在する。
# 関連ファイル: 定義書.md, plans.md, README.md

"""MNISTのフルバッチとミニバッチで計算時間を比較するベンチマーク。"""

import time
from typing import Tuple

import numpy as np

try:
    from sklearn.datasets import fetch_openml  # type: ignore
except Exception:
    fetch_openml = None

try:
    from dataset.mnist import load_mnist  # type: ignore
except Exception:
    load_mnist = None


def softmax(x: np.ndarray) -> np.ndarray:
    """数値安定化付きソフトマックス。"""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """バッチ対応の交差エントロピー。"""
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    batch_size = y.shape[0]
    epsilon = 1e-7  # ゼロ割り防止の微小値
    return float(-np.sum(t * np.log(y + epsilon)) / batch_size)


def forward_once(x: np.ndarray, t: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    """1ステップ分の順伝播と損失計算。"""
    logits = np.dot(x, W) + b
    probs = softmax(logits)
    return cross_entropy_error(probs, t)


def measure_pass(x: np.ndarray, t: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """順伝播の処理時間（秒）と損失を返す。"""
    start = time.time()
    loss = forward_once(x, t, W, b)
    end = time.time()
    return end - start, loss


def load_data() -> Tuple[np.ndarray, np.ndarray, str]:
    """スキル順でデータを読み込む。sklearn MNIST→書籍版→ダミー。"""
    if fetch_openml:
        try:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
            x = mnist["data"].astype(np.float32) / 255.0
            labels = mnist["target"].astype(np.int64)
            t = np.eye(10, dtype=np.float32)[labels]
            return x, t, "sklearn-mnist"
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: sklearn MNIST取得に失敗しました。syntheticにフォールバックします ({exc})")

    if load_mnist:
        (x_train, t_train), _ = load_mnist(
            normalize=True, flatten=True, one_hot_label=True
        )
        return x_train.astype(np.float32), t_train.astype(np.float32), "book-mnist"

    x_dummy = np.random.rand(60000, 784).astype(np.float32)
    t_dummy = np.eye(10, dtype=np.float32)[np.random.randint(0, 10, 60000)]
    return x_dummy, t_dummy, "synthetic"


def main() -> None:
    x_train, t_train, data_source = load_data()

    np.random.seed(42)
    W = np.random.randn(784, 10) * 0.01
    b = np.zeros(10, dtype=float)

    runs = 5
    batch_size = 100
    full_times, mini_times = [], []
    full_losses, mini_losses = [], []

    for _ in range(runs):
        full_t, full_l = measure_pass(x_train, t_train, W, b)
        full_times.append(full_t)
        full_losses.append(full_l)

        batch_indices = np.random.choice(x_train.shape[0], batch_size, replace=False)
        x_batch = x_train[batch_indices]
        t_batch = t_train[batch_indices]
        mini_t, mini_l = measure_pass(x_batch, t_batch, W, b)
        mini_times.append(mini_t)
        mini_losses.append(mini_l)

    avg_full = float(np.mean(full_times))
    avg_mini = float(np.mean(mini_times))
    std_full = float(np.std(full_times))
    std_mini = float(np.std(mini_times))
    speedup = avg_full / avg_mini if avg_mini > 0 else float("inf")

    print("--- Benchmark Result (mean of runs) ---")
    print(f"Runs                   : {runs}")
    print(f"Full Batch (60000 data): {avg_full:.4f} sec | last loss={full_losses[-1]:.4f}")
    print(f"Mini Batch ({batch_size} data) : {avg_mini:.4f} sec | last loss={mini_losses[-1]:.4f}")
    print(f"Speedup Factor         : {speedup:.2f} times faster")
    print(f"Data Source            : {data_source}")
    print(f"Full times per run     : {[round(t,4) for t in full_times]}")
    print(f"Mini times per run     : {[round(t,4) for t in mini_times]}")

    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-paper")

        labels = ["Full Batch", "Mini Batch"]
        means = [avg_full, avg_mini]
        stds = [std_full, std_mini]

        fig, ax = plt.subplots(figsize=(7, 4.5))

        bars = ax.bar(
            labels,
            means,
            yerr=stds,
            capsize=6,
            color=["#4c72b0", "#55a868"],
            alpha=0.8,
            edgecolor="#1f1f1f",
        )

        # 個別計測値（スキャッタ）
        rng = np.random.default_rng(0)
        jitters_full = rng.uniform(-0.08, 0.08, size=runs)
        jitters_mini = rng.uniform(-0.08, 0.08, size=runs)
        ax.scatter(np.full(runs, 0) + jitters_full, full_times, color="#1f4b99", s=28, alpha=0.9, label="Full runs")
        ax.scatter(np.full(runs, 1) + jitters_mini, mini_times, color="#2b8c55", s=28, alpha=0.9, label="Mini runs")

        ax.set_ylabel("Time (sec)")
        ax.set_title(f"Forward Pass Time ({runs} runs)")
        ax.set_yscale("log")
        ax.yaxis.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_ylim(bottom=min(mini_times) * 0.5)

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean * 1.15,
                f"{mean:.4f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#1f1f1f",
            )

        ax.text(
            0.5,
            0.95,
            f"Speedup ≈ {speedup:.1f}x",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#444", alpha=0.9),
        )

        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig("benchmark_plot.png", dpi=300)
        print("Plot saved: benchmark_plot.png (log-scale, mean±std, per-run dots)")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: plot generation failed ({exc})")


if __name__ == "__main__":
    main()
