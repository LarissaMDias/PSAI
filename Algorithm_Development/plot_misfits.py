#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:19:55 2026

@author: larissadias
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_misfits(
    y: pd.DataFrame,
    *,
    targets: Iterable[str] | None = None,
    output_dir: str | Path | None = None,
    bins: int = 50,
    show: bool = True,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot calculated misfit distributions and return summary statistics.

    ``y`` should be the target DataFrame returned by ``model_df_create``.
    Misfits are assumed to use the convention model minus observation.
    """
    if not isinstance(y, pd.DataFrame) or y.empty:
        raise ValueError("y must be a non-empty pandas DataFrame")

    selected = list(targets) if targets is not None else list(y.columns)
    missing = sorted(set(selected) - set(y.columns))
    if missing:
        raise KeyError(f"Targets not found in y: {missing}")
    if not selected:
        raise ValueError("No targets selected")

    summary = pd.DataFrame({
        "n": y[selected].notna().sum(),
        "missing": y[selected].isna().sum(),
        "mean": y[selected].mean(),
        "median": y[selected].median(),
        "std": y[selected].std(),
        "min": y[selected].min(),
        "max": y[selected].max(),
    })
    print(summary.to_string())

    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, target in zip(axes, selected):
        values = pd.to_numeric(y[target], errors="coerce").dropna()
        ax.hist(values, bins=bins, color="#377eb8", alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(values.mean(), color="#d62728", linewidth=2, label=f"mean = {values.mean():.3g}")
        ax.set_title(target)
        ax.set_xlabel("Misfit (model − observation)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    for ax in axes[len(selected):]:
        ax.remove()

    fig.suptitle("Calculated misfit distributions", fontsize=14)
    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "misfit_summary.csv")
        fig.savefig(output_dir / "misfit_distributions.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return summary, fig


if __name__ == "__main__":
    print("Import plot_misfits() and call it with y from model_df_create().")
