# src/utils/splits.py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

def make_holdout_and_inner_folds(df, y_col="target", group_col="patient_id",
                                 test_size=0.2, seed=42):
    df = df.copy()

    # --- Holdout TEST, grouped ---
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col].values
    y = df[y_col].values
    trainval_idx, test_idx = next(gss.split(df, groups=groups))
    df["split"] = "trainval"
    df.loc[test_idx, "split"] = "test"
    df["inner_fold"] = -1  # Initialize for all rows

    # --- Inner 5-fold on trainval, grouped + stratified ---
    inner = df[df["split"] == "trainval"].copy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, val_index) in enumerate(sgkf.split(inner, y=inner[y_col].values,
                                                     groups=inner[group_col].values)):
        inner.iloc[val_index, inner.columns.get_loc("inner_fold")] = fold

    df.loc[inner.index, "inner_fold"] = inner["inner_fold"]
    return df
