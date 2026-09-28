#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:15:21 2026

@author: lara
"""

from __future__ import annotations

import pandas as pd

from xgb_single_target_cv import run_xgb_cv


def run_all_hypotheses(
    cv_splits,
    *,
    target: str,
    method: str,
    hypotheses: tuple[str, ...] = ("A", "0"),
    tune: bool = True,
    n_iter: int = 20,
):
    """Evaluate all specified feature hypotheses for one target and CV method."""
    results = []
    predictions = {}
    models = {}

    for hypothesis in hypotheses:
        fold_results, pred, fitted = run_xgb_cv(
            cv_splits,
            target=target,
            method=method,
            hypothesis=hypothesis,
            tune=tune,
            n_iter=n_iter,
        )
        results.append(fold_results)
        predictions[hypothesis] = pred
        models[hypothesis] = fitted

    return pd.concat(results, ignore_index=True), predictions, models


# Example:
# all_results, predictions, models = run_all_hypotheses(
#     cv_splits,
#     target="TA_misfit",
#     method="year",
#     hypotheses=("A", "0"),
#     tune=True,
#     n_iter=20,
# )
