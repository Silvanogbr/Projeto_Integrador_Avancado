"""Temporal preparation and supervised model training for AeroMotion data."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

SENSORS = ["temperatura_mancal_c", "vibracao_rms_mms", "pressao_oleo_bar",
           "rotacao_rpm", "emissao_acustica_db", "corrente_eletrica_amp"]
PHYSICAL = {"temperatura_mancal_c": (20, 180), "vibracao_rms_mms": (0, 20),
            "pressao_oleo_bar": (0, 8), "rotacao_rpm": (1800, 4000),
            "emissao_acustica_db": (20, 100), "corrente_eletrica_amp": (0, 50)}
TARGETS = ["Normal", "Desgaste_Rolamento", "Desbalanceamento_Rotor", "Cavitacao_Superaquecimento"]


def load_and_engineer(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load raw CSV, flag invalid extremes, and make strictly causal rolling features."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["id_equipamento", "timestamp"], kind="stable").reset_index(drop=True)
    if df.duplicated(["id_equipamento", "timestamp"]).any():
        raise ValueError("Duplicated equipment/timestamp keys")
    intervals = df.timestamp.drop_duplicates().sort_values().diff().dropna()
    equipment_intervals = df.groupby("id_equipamento").timestamp.diff().dropna()
    equipment_interval_summary = {
        equipment: values.dt.total_seconds().div(60).describe().to_dict()
        for equipment, values in equipment_intervals.groupby(df.loc[equipment_intervals.index, "id_equipamento"])
    }
    audit = {"rows": len(df), "columns": list(df.columns), "missing_count": df.isna().sum().to_dict(),
             "missing_rate": df.isna().mean().to_dict(), "global_interval_minutes": intervals.dt.total_seconds().div(60).describe().to_dict(),
             "equipment_interval_minutes": equipment_interval_summary,
             "equipment_missing_rate": df.groupby("id_equipamento")[SENSORS].apply(lambda x: x.isna().mean()).to_dict("index"),
             "equipment": sorted(df.id_equipamento.unique().tolist()), "target_counts": df.tipo_falha.value_counts().sort_index().to_dict()}
    for col, (low, high) in PHYSICAL.items():
        # Limits are intentionally broad engineering plausibility bounds; the CSV dictionary's
        # normal operating ranges are narrower and should not censor genuine fault signatures.
        df.loc[~df[col].between(low, high), col] = np.nan
    for col in ["vibracao_rms_mms", "temperatura_mancal_c"]:
        for _, indexes in df.groupby("id_equipamento", sort=False).groups.items():
            part = df.loc[indexes].sort_values("timestamp")
            past = part.set_index("timestamp")[col].shift(1)
            for label, window in [("1h", "1h"), ("6h", "6h")]:
                # Time windows respect the actual per-equipment sampling cadence.
                df.loc[part.index, f"{col}_rolling_mean_{label}"] = past.rolling(window, min_periods=1).mean().to_numpy()
                df.loc[part.index, f"{col}_rolling_std_{label}"] = past.rolling(window, min_periods=2).std().to_numpy()
    df["severidade_dinamica"] = df.vibracao_rms_mms / df.rotacao_rpm.replace(0, np.nan)
    df["hora_dia"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    df["dia_semana"] = df.timestamp.dt.dayofweek
    audit["invalid_extremes_replaced"] = int(df[SENSORS].isna().sum().sum()) - sum(audit["missing_count"][c] for c in SENSORS)
    return df, audit


def _preprocessor(feature_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                                       ("scale", RobustScaler())]), num_cols),
                              ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                         ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])


def _interpolate_partition(frame: pd.DataFrame) -> pd.DataFrame:
    """Interpolate short internal sensor gaps by equipment without crossing split boundary."""
    frame = frame.copy()
    # Interpolate raw sensor measurements only. Rolling features are causal outputs and any
    # remaining gaps are left for the train-fitted median imputer.
    numeric = [c for c in frame.columns if c in SENSORS]
    for _, indexes in frame.groupby("id_equipamento", sort=False).groups.items():
        part = frame.loc[indexes].sort_values("timestamp")
        row_indexes = part.index
        part = part.set_index("timestamp")
        for col in numeric:
            part[col] = part[col].interpolate(method="time", limit=3, limit_area="inside")
        frame.loc[row_indexes, numeric] = part[numeric].to_numpy()
    return frame


def train(path: str | Path, out_dir: str | Path = "artifacts") -> dict:
    """Chronological 70/30 holdout; class-weighted models and normal-only anomaly detector."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df, audit = load_and_engineer(path)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split = int(len(df) * .70)
    tr_raw, te_raw = df.iloc[:split].copy(), df.iloc[split:].copy()
    # Interpolate each evaluation block independently. Cross-validation uses the raw block so
    # interpolation cannot use a future fold's observations to fill an earlier fold.
    tr, te = _interpolate_partition(tr_raw), _interpolate_partition(te_raw)
    audit["split"] = {"train_rows": len(tr), "test_rows": len(te), "train_end": str(tr.timestamp.max()), "test_start": str(te.timestamp.min())}
    excluded = {"timestamp", "tipo_falha", "horas_operacao_acumuladas"}
    features = [c for c in df.columns if c not in excluded]
    cats = ["id_equipamento", "modo_operacao"]
    models = {
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
    }
    result, fitted = {}, {}
    Xtr, Xte, Xtr_cv = tr[features], te[features], tr_raw[features]
    ytr, yte = tr.tipo_falha, te.tipo_falha
    def score(model, extra=None):
        pred = model.predict(Xte)
        values = {"accuracy": accuracy_score(yte, pred),
                  "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
                  "precision_weighted": precision_score(yte, pred, average="weighted", zero_division=0),
                  "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
                  "recall_weighted": recall_score(yte, pred, average="weighted", zero_division=0),
                  "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
                  "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
                  "confusion_matrix": confusion_matrix(yte, pred, labels=[0,1,2,3]).tolist(),
                  "classification_report": classification_report(yte, pred, labels=[0,1,2,3], target_names=TARGETS, output_dict=True, zero_division=0)}
        if extra: values.update(extra)
        return values

    # Compare and tune both model families using forward-chaining folds inside the training
    # period. Model selection uses validation recall_macro, never the final temporal test set.
    class_weights = ["balanced", {0: 1, 1: 2, 2: 2, 3: 2}, {0: 1, 1: 3, 2: 2, 3: 2}]
    cv_scores = {}
    for name, model_class in models.items():
        estimator = Pipeline([("prep", _preprocessor(features, cats)),
                              ("model", model_class(n_jobs=-1, random_state=42))])
        search = RandomizedSearchCV(estimator,
            {"model__n_estimators": [250, 350], "model__max_depth": [None, 18, 28],
             "model__min_samples_leaf": [1, 2, 4], "model__class_weight": class_weights},
            n_iter=6, scoring="recall_macro", cv=TimeSeriesSplit(n_splits=3),
            random_state=42, n_jobs=1, refit=False)
        search.fit(Xtr_cv, ytr)
        key = f"{name}_tuned"
        cv_scores[key] = float(search.best_score_)
        final_estimator = Pipeline([("prep", _preprocessor(features, cats)),
                                    ("model", model_class(n_jobs=-1, random_state=42,
                                                           **{k.replace('model__', ''): v for k, v in search.best_params_.items()}))])
        final_estimator.fit(Xtr, ytr)
        result[key] = score(final_estimator, {"best_params": search.best_params_,
                                                       "cv_recall_macro": cv_scores[key]})
        fitted[key] = final_estimator
    best = max(cv_scores, key=cv_scores.get)
    joblib.dump(fitted[best], out / "model.joblib")
    normal = tr[tr.tipo_falha == 0]
    anomaly = Pipeline([("prep", _preprocessor(features, cats)),
                        ("model", IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1))])
    anomaly.fit(normal[features]); joblib.dump(anomaly, out / "isolation_forest.joblib")
    manifest = {"model": best, "features": features, "target_names": TARGETS, "metrics": result, "data_audit": audit,
                "anomaly_train_rows": len(normal), "note": "Operational validation only; labels and acquisition provenance are assumed accurate."}
    (out / "metrics.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("csv"); parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(); print(json.dumps(train(args.csv, args.out), indent=2, ensure_ascii=False, default=str))
