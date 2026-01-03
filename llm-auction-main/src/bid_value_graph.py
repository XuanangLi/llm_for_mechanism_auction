import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent

# input files
FILES = [
    "gpt4o-mini_results/SEAL/seal_first_private/result_10_2025-12-21_11-18-36-293805.json",
    "gpt4o-mini_results/SEAL/seal_first_private/result_10_2025-12-21_11-23-20-690396.json",
    "gpt4o-mini_results/SEAL/seal_first_private/result_10_2025-12-21_11-26-55-472134.json",
]



# 重要：value 列表与 agent 的固定对应顺序（从你 JSON 的 profit/winner 可验证）
AGENTS = ["Bidder Andy", "Bidder Betty", "Bidder Charles"]

def load_one_run(json_path: str, run_id: int) -> pd.DataFrame:
    """
    从单个 JSON 文件抽取 (value, bid) 点云。
    输出列：run_id, round, agent, value, bid
    """
    json_path = str(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for key, rd in data.items():
        if not key.startswith("round_"):
            continue

        r = int(rd["round"])
        values = rd["value"]  # list, len = n_agents

        # bidding history: list[dict] -> dict agent->bid
        bh = rd["history"]["bidding history"]
        bid_map = {x["agent"]: float(x["bid"]) for x in bh}

        # 逐 agent 对齐 value[i] 与 bid_map[agent]
        for i, agent in enumerate(AGENTS):
            if agent not in bid_map:
                raise KeyError(f"Agent {agent} not found in bidding history in {json_path}, round={r}")

            rows.append({
                "run_id": run_id,
                "round": r,
                "agent": agent,
                "value": float(values[i]),
                "bid": float(bid_map[agent]),
                "file": Path(json_path).name,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["run_id", "round", "agent"]).reset_index(drop=True)

def load_all_runs(files) -> pd.DataFrame:
    dfs = []
    for idx, fp in enumerate(files, start=1):
        dfs.append(load_one_run(fp, run_id=idx))
    return pd.concat(dfs, ignore_index=True)

def plot_value_vs_bid(
    df: pd.DataFrame,
    title: str = "Assigned Value vs Bid (3 runs)",
    theory_line: str = "bid_equals_value",
    lowess_frac: float = 0.25,
    pooled_lowess: bool = True,
    per_run_lowess: bool = False,
):
    """
    画散点 + 理论线 + LOWESS 平滑线
    - pooled_lowess=True: 用所有点画一条红色平滑线（最像你参考图）
    - per_run_lowess=True: 每个 run 各画一条平滑线（便于看三次实验稳定性）
    """
    plt.figure(figsize=(7.5, 5.5))

    # 1) 点云：每个 run 用不同 marker
    markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}
    for run_id, sub in df.groupby("run_id"):
        plt.scatter(
            sub["value"], sub["bid"],
            alpha=0.45,
            s=30,
            marker=markers.get(run_id, "o"),
            label=f"Run {run_id}"
        )

    # 2) 黑色虚线：理论基准
    xmin = float(df["value"].min())
    xmax = float(df["value"].max())
    xx = np.linspace(xmin, xmax, 200)

    if theory_line == "bid_equals_value":
        plt.plot(xx, xx, "k--", linewidth=2.0, label="Theory: bid = value")
    else:
        # 如果你未来要画 FPSB（n=3 的风险中性 BNE）
        # theory_line="fpsb_bne", 则画 b(v)=(n-1)/n v
        n = len(AGENTS)
        plt.plot(xx, (n - 1) / n * xx, "k--", linewidth=2.0, label=f"Theory: bid={(n-1)}/{n}·value")

    # 3) 红色曲线：LOWESS 平滑趋势
    def add_lowess(x, y, label, lw=2.5):
        sm = lowess(y, x, frac=lowess_frac, return_sorted=True)
        plt.plot(sm[:, 0], sm[:, 1], linewidth=lw, label=label)

    if pooled_lowess:
        add_lowess(df["value"].to_numpy(), df["bid"].to_numpy(), label="LOWESS (pooled)")

    if per_run_lowess:
        for run_id, sub in df.groupby("run_id"):
            add_lowess(
                sub["value"].to_numpy(),
                sub["bid"].to_numpy(),
                label=f"LOWESS (Run {run_id})",
                lw=2.0
            )

    plt.title(title)
    plt.xlabel("Assigned value for the good")
    plt.ylabel("Bid")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_all = load_all_runs(FILES)
    print(df_all.head(10))
    # 画你参考图那种：一条 pooled 的红色平滑线 + 理论 bid=value
    plot_value_vs_bid(
        df_all,
        title="Second-Price/Clock Style: Assigned Value vs Bid (3 runs)",
        theory_line="bid_equals_value",
        lowess_frac=0.25,
        pooled_lowess=True,
        per_run_lowess=False,
    )

    # 如果你也想看每次实验是否一致，可把 per_run_lowess=True
    # plot_value_vs_bid(df_all, per_run_lowess=True, pooled_lowess=False)
