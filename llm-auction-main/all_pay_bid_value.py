from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


PROJECT_ROOT = Path(__file__).resolve().parent

FILES = [
    "gpt4o-mini_results/SEAL/seal_all_pay_private/result_10_2025-12-21_12-19-58-280238.json",
    "gpt4o-mini_results/SEAL/seal_all_pay_private/result_10_2025-12-21_12-24-11-233660.json",
    "gpt4o-mini_results/SEAL/seal_all_pay_private/result_10_2025-12-21_12-28-27-573423.json",
]

# agent order
AGENTS = ["Bidder Andy", "Bidder Betty", "Bidder Charles"]


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

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    dfs = []
    for idx, p in enumerate(paths, start=1):
        dfs.append(load_one_run(p, run_id=idx))
    return pd.concat(dfs, ignore_index=True)


def plot_first_price_value_vs_bid(
    df: pd.DataFrame,
    lowess_frac: float = 0.25,
    n_bidders: int = 3,
    save_path: Path | None = None,
):
    x = df["value"].to_numpy()
    y = df["bid"].to_numpy()

    plt.figure(figsize=(8, 5.8))

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

    xmin, xmax = float(np.min(x)), float(np.max(x))
    xx = np.linspace(xmin, xmax, 300)

    plt.plot(xx, xx, "k--", linewidth=2.2, label="Actual Value")

    bne = (n_bidders - 1) / n_bidders * (xx ** n_bidders) / (V ** (n_bidders - 1))
    plt.plot(
        xx, bne,
        linestyle="--",
        linewidth=2.2,
        label=rf"Bayes-Nash Equilibrium Strategy: $\frac{{{n_bidders-1}}}{{{n_bidders}\cdot {int(V)}^{{{n_bidders-1}}}}}v^{{{n_bidders}}}$"
    )

    sm = lowess(y, x, frac=lowess_frac, return_sorted=True)
    plt.plot(sm[:, 0], sm[:, 1], linewidth=2.8, label="Smoothed Data")

    plt.title("All-Pay: Assigned Value vs. Bid")
    plt.xlabel("Assigned value for the good")
    plt.ylabel("LLM agent's bid")

    plt.legend(loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()


if __name__ == "__main__":
    df_all = load_all_runs(PROJECT_ROOT, FILES)
    print(df_all.head())

    n_bidders = len(AGENTS)  

    # Actual value benchmark
    df_all["actual_value"] = df_all["value"]

    # All-Pay Bayes–Nash equilibrium (IPV, risk-neutral, U[0,99])
    V_MAX = 99.0
    df_all["bne_bid"] = (2.0 / (3.0 * (V_MAX ** 2))) * (df_all["value"] ** 3)

    df_all["abs_dev_actual"] = (df_all["bid"] - df_all["actual_value"]).abs()
    df_all["abs_dev_bne"] = (df_all["bid"] - df_all["bne_bid"]).abs()

    print("\n=== All-Pay Absolute Deviation Summary (Pooled) ===")
    print(df_all[["abs_dev_actual", "abs_dev_bne"]].describe())

    print("\n=== All-Pay Absolute Deviation by Run ===")
    print(
        df_all
        .groupby("run_id")[["abs_dev_actual", "abs_dev_bne"]]
        .mean()
    )

    plot_first_price_value_vs_bid(
        df_all,
        lowess_frac=0.25,       
        n_bidders=len(AGENTS),  
        save_path=PROJECT_ROOT / "plots" / "gpt4o-mini" / "appsb.png",
    )
