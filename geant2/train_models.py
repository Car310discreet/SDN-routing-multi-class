"""
train_models.py - train and compare the 4-class traffic classifiers on the GEANT2 dataset.

Classes: 0 Extremely Low, 1 Low, 2 High, 3 Extremely High. Split 70 % train / 15 % validation /
15 % test (stratified, as in the baseline).

By default the four models most likely to do best on this data are trained (`--models top4`):
    XGBoost, Gradient Boosting (boosting), Random Forest (bagging), SVM (RBF kernel)
The class boundaries are non-linear (RTT and loss jump once a link saturates), which favours tree
ensembles and kernel methods. `--models all` adds Logistic Regression, Decision Tree, Bagging and
AdaBoost.
The best model is the one with the highest validation macro-F1 (the test set is only reported). It
is saved with its feature list to geant2/models/best_model.joblib for controller_routing.py.

Outputs in geant2/results/training/:
    model_comparison.csv   test accuracy, macro precision / recall / F1, validation F1, timings
    confusion_matrix.csv   best model, test set (rows = actual, columns = predicted)
    model_comparison.png   accuracy, precision and F1 per model

    .venv/bin/python geant2/train_models.py                        # all features
    .venv/bin/python geant2/train_models.py --feature-set active   # end-to-end features only
    .venv/bin/python geant2/train_models.py --models all           # all 8 classifiers
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_classifier import DEFAULT_MODEL, FEATURE_SETS  # noqa: E402
from traffic_gen import CLASS_NAMES  # noqa: E402

log = logging.getLogger("train_models")
GEANT2_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(GEANT2_DIR, "data", "geant2_dataset.csv")
DEFAULT_RESULTS = os.path.join(GEANT2_DIR, "results", "training")
SEED = 40
TOP4 = ["XGBoost", "Random Forest", "Gradient Boosting", "SVM"]


def build_models(names: list[str] | None = None) -> dict:
    # Distance / margin based models need scaled features (RTT in ms vs utilisation in %)
    scaled = lambda model: make_pipeline(StandardScaler(), model)  # noqa: E731
    models = {
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        # SVC(probability=True) is deprecated in scikit-learn 1.9; calibrate explicitly instead
        "SVM": scaled(CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
                                             ensemble=False)),
        "XGBoost": xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", early_stopping_rounds=10,
                                     n_estimators=500, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(random_state=SEED), n_estimators=100, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    return models if names is None else {name: models[name] for name in names}


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def load_dataset(path: str, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path} (run geant2/collect_dataset.py first)")
    df = pd.read_csv(path)
    missing = [c for c in features + ["traffic_class"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    df = df.dropna(subset=features + ["traffic_class"])
    y = df["traffic_class"].astype(int).to_numpy()
    present = sorted(set(y))
    if present != list(range(len(CLASS_NAMES))):
        raise ValueError(f"Dataset must contain all 4 classes 0-3, has {present}")
    return df[features].to_numpy(dtype=float), y


def plot_comparison(table: pd.DataFrame, best: str, path: str, note: str | None = None) -> None:
    sys.path.append(os.path.join(os.path.dirname(GEANT2_DIR), "baseline"))
    import matplotlib.pyplot as plt
    from visualize_benchmarks import COLORS, INK, INK_2, MUTED, apply_style

    apply_style()
    order = table.sort_values("f1")["model"].tolist()
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision (macro)"), ("f1", "F1 (macro)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.42 * len(order) + 1.4), sharey=True)
    for ax, (column, title) in zip(axes, metrics):
        values = table.set_index("model").loc[order, column]
        colors = [COLORS["hop_count"] if m == best else "#c3c2b7" for m in order]
        ax.barh(order, values, color=colors, height=0.62, edgecolor="none")
        for y, value in enumerate(values):
            ax.annotate(f"{value:.3f}", xy=(value, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK if order[y] == best else INK_2)
        ax.set_xlim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(note or f"Test-set scores; best on validation F1: {best} (blue)", x=0.01, ha="left",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(args) -> str:
    features = FEATURE_SETS[args.feature_set]
    X, y = load_dataset(args.data, features)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    log.info("%d rows, %d features (%s); train %d / validation %d / test %d", len(y), len(features),
             args.feature_set, len(y_train), len(y_val), len(y_test))

    rows, fitted = [], {}
    names = None if args.models == "all" else TOP4
    for name, model in build_models(names).items():
        start = time.perf_counter()
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        train_s = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_ms = (time.perf_counter() - start) * 1000 / len(X_test)
        val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
        rows.append({"model": name, **scores(y_test, y_pred), "val_f1": val_f1, "train_s": train_s,
                     "infer_ms_per_path": infer_ms})
        fitted[name] = (model, y_pred)

    table = pd.DataFrame(rows)
    best = table.sort_values(["val_f1", "infer_ms_per_path"], ascending=[False, True]).iloc[0]["model"]
    print(f"\n{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Val F1':>9}{'ms/path':>9}")
    for _, r in table.iterrows():
        mark = "  <- best" if r["model"] == best else ""
        print(f"{r['model']:<22}{r['accuracy']:>10.3f}{r['precision']:>11.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}"
              f"{r['val_f1']:>9.3f}{r['infer_ms_per_path']:>9.3f}{mark}")

    model, y_pred = fitted[best]
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    print(f"\n{best} confusion matrix on the test set (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    os.makedirs(args.results, exist_ok=True)
    table.round(6).to_csv(os.path.join(args.results, "model_comparison.csv"), index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(os.path.join(args.results, "confusion_matrix.csv"))
    plot_comparison(table, best, os.path.join(args.results, "model_comparison.png"))

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    metrics = table.set_index("model").loc[best].round(4).to_dict()
    joblib.dump({"model": model, "name": best, "features": features, "classes": CLASS_NAMES,
                 "feature_set": args.feature_set, "metrics": metrics}, args.model_out)
    with open(os.path.join(args.results, "best_model.json"), "w") as handle:
        json.dump({"name": best, "feature_set": args.feature_set, "features": features, "metrics": metrics},
                  handle, indent=2)
    print(f"\nSaved {best} to {args.model_out}; comparison in {args.results}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="all")
    parser.add_argument("--models", choices=["top4", "all"], default="top4")
    parser.add_argument("--model-out", default=DEFAULT_MODEL)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        train(args)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
