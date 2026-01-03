from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# -----------------------
# 1) 路径与文件
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent

FILES = [
    "gemini2.5flash_results/seal_first_private/result_10_2025-12-30_15-31-40-114606.json", #"gpt4o-mini_results/SEAL/seal_first_private/result_10_2025-12-21_11-18-36-293805.json",
    "gemini2.5flash_results/seal_first_private/result_10_2025-12-30_15-42-16-561406.json", #"gpt4o-mini_results/SEAL/seal_first_private/result_10_2025-12-21_11-23-20-690396.json",
    "gemini2.5flash_results/seal_first_private/result_10_2025-12-30_15-54-04-824468.json", #"gpt4o-mini_results/SEAL/seal_first_private/result_10_2025-12-21_11-26-55-472134.json",
]

# agent order
AGENTS = ["Bidder Andy", "Bidder Betty", "Bidder Charles"]

# -----------------------
# 2) 读 JSON 并抽取点云 (value, bid)
# -----------------------
def load_one_run(json_path: Path, run_id: int) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for key, rd in data.items():
        if not key.startswith("round_"):
            continue

        r = int(rd["round"])
        values = rd["value"]  # list: [v_Andy, v_Betty, v_Charles]

        # bidding history: list[{"agent":..., "bid":...}]
        bh = rd["history"]["bidding history"]
        bid_map = {x["agent"]: float(x["bid"]) for x in bh}

        for i, agent in enumerate(AGENTS):
            if agent not in bid_map:
                raise KeyError(f"Missing {agent} in bidding history: file={json_path.name}, round={r}")
            rows.append(
                {
                    "run_id": run_id,
                    "round": r,
                    "agent": agent,
                    "value": float(values[i]),
                    "bid": float(bid_map[agent]),
                    "file": json_path.name,
                }
            )

    return pd.DataFrame(rows).sort_values(["run_id", "round", "agent"]).reset_index(drop=True)


def load_all_runs(project_root: Path, files: list[str]) -> pd.DataFrame:
    paths = [project_root / fp for fp in files]

    # 文件存在性检查
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    dfs = []
    for idx, p in enumerate(paths, start=1):
        dfs.append(load_one_run(p, run_id=idx))
    return pd.concat(dfs, ignore_index=True)


# -----------------------
# 3) 画 First-Price 图：点云 + y=x + BNE + LOWESS
# -----------------------
def plot_first_price_value_vs_bid(
    df: pd.DataFrame,
    lowess_frac: float = 0.25,
    n_bidders: int = 3,
    save_path: Path | None = None,
):
    x = df["value"].to_numpy()
    y = df["bid"].to_numpy()

    # 画布
    plt.figure(figsize=(8, 5.8))

    # 点云：可以按 run 叠加，也可以 pooled 一起画
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for run_id, sub in df.groupby("run_id"):
        plt.scatter(
            sub["value"],
            sub["bid"],
            alpha=0.45,
            s=28,
            marker=markers.get(run_id, "o"),
            label=f"Bids (Run {run_id})",
        )

    # 范围
    xmin, xmax = float(np.min(x)), float(np.max(x))
    xx = np.linspace(xmin, xmax, 300)

    # 黑色虚线：y = x （Actual Value）
    plt.plot(xx, xx, "k--", linewidth=2.2, label="Actual Value")

    # 绿色虚线：First-price risk-neutral BNE: (n-1)/n * v
    bne = (n_bidders - 1) / n_bidders * xx
    plt.plot(xx, bne, linestyle="--", linewidth=2.2, label=r"Bayes-Nash Equilibrium Strategy: $\frac{n-1}{n}v$")

    # 红色 LOWESS 平滑
    sm = lowess(y, x, frac=lowess_frac, return_sorted=True)
    plt.plot(sm[:, 0], sm[:, 1], linewidth=2.8, label="Smoothed Data")

    # 标题与轴
    plt.title("First-Price: Assigned Value vs. Bid")
    plt.xlabel("Assigned value for the good")
    plt.ylabel("LLM agent's bid")

    # 图例
    plt.legend(loc="upper left")
    plt.tight_layout()

    # 保存
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()


if __name__ == "__main__":
    df_all = load_all_runs(PROJECT_ROOT, FILES)
    print(df_all.head())

    # =========================
    # 1️⃣ 理论 benchmark
    # =========================
    n_bidders = len(AGENTS)

    # Actual value benchmark
    df_all["actual_value"] = df_all["value"]

    # FPSB Bayes–Nash equilibrium
    df_all["bne_bid"] = (n_bidders - 1) / n_bidders * df_all["value"]

    # =========================
    # 2️⃣ 绝对偏差
    # =========================
    df_all["abs_dev_actual"] = (df_all["bid"] - df_all["actual_value"]).abs()
    df_all["abs_dev_bne"] = (df_all["bid"] - df_all["bne_bid"]).abs()

    # =========================
    # 3️⃣ 打印结果（整体）
    # =========================
    print("\n=== Absolute Deviation Summary (Pooled) ===")
    print(df_all[["abs_dev_actual", "abs_dev_bne"]].describe())

    # =========================
    # 4️⃣ 可选：按 run 打印
    # =========================
    print("\n=== Absolute Deviation by Run ===")
    print(
        df_all
        .groupby("run_id")[["abs_dev_actual", "abs_dev_bne"]]
        .mean()
    )

    plot_first_price_value_vs_bid(
        df_all,
        lowess_frac=0.25,       # 越大越平滑；0.2~0.35 一般都不错
        n_bidders=len(AGENTS),  # 这里是 3
        save_path=PROJECT_ROOT / "plots" / "gemini2.5flash" / "fpsb.png",
    )
