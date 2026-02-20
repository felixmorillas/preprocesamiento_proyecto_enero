
import re
import joblib
import numpy as np
import pandas as pd
import os

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from tqdm.auto import tqdm


class Training:

    # ==============================
    # VALIDACIÓN CRUZADA
    # ==============================
    @classmethod
    def run_cross_validation(
        cls,
        MODEL_VARIANTS: dict,
        preprocessor,
        X_train,
        y_train,
        *,
        cv_splits: int = 5,
        cv_shuffle: bool = True,
        cv_random_state: int = 42,
        use_tqdm: bool = False
    ):

        # Columnas (orden fijo)
        FOLDS_COLS = [
            "model","fold","threshold","clf_params",
            "accuracy","recall","precision","specificity","f1","roc_auc",
            "tn","fp","fn","tp",
        ]
        SUMMARY_COLS = [
            "model",
            "precision_mean","precision_std",
            "roc_auc_mean","roc_auc_std",
            "recall_mean","recall_std",
            "f1_mean","f1_std",
            "specificity_mean","specificity_std",
            "accuracy_mean","accuracy_std",
            "threshold","clf_params",
        ]

        # Splitter CV
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=cv_random_state)

        # Alinear índices
        Xtr = X_train.reset_index(drop=True)
        ytr = pd.Series(y_train).reset_index(drop=True)

        rows = []

        items = MODEL_VARIANTS.items()
        if use_tqdm:
            items = tqdm(items, desc="Iterando modelos")

        for model_id, spec in items:
            base_estimator = spec["estimator"]
            thr = float(spec.get("threshold", 0.50))

            for fold, (idx_tr, idx_va) in enumerate(skf.split(Xtr, ytr), start=1):
                X_fold_tr = Xtr.iloc[idx_tr]
                y_fold_tr = ytr.iloc[idx_tr]
                X_fold_va = Xtr.iloc[idx_va]
                y_fold_va = ytr.iloc[idx_va]

                # Pipeline por fold (todo dentro de Training, no en el notebook)
                pipe = Pipeline([
                    ("prep", clone(preprocessor)),
                    ("clf", clone(base_estimator)),
                ])

                m = cls.eval_one_fold(pipe, X_fold_tr, y_fold_tr, X_fold_va, y_fold_va, threshold=thr)

                row = {c: np.nan for c in FOLDS_COLS}
                row.update({
                    "model": model_id,
                    "threshold": thr,
                    "clf_params": spec.get("params"),
                    "fold": fold
                })
                row.update(m)
                rows.append(row)

        cv_folds_df = pd.DataFrame(rows, columns=FOLDS_COLS)

        cv_summary_df = (
            cv_folds_df
            .groupby("model", as_index=False)
            .agg(
                roc_auc_mean=("roc_auc", "mean"),
                roc_auc_std=("roc_auc", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                specificity_mean=("specificity", "mean"),
                specificity_std=("specificity", "std"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                threshold=("threshold", "first")
            )
            .sort_values(by=["precision_mean", "roc_auc_mean", "f1_mean"], ascending=False)
        ).reindex(columns=SUMMARY_COLS)

        cv_folds_df   = cv_folds_df.round(4)
        cv_summary_df = cv_summary_df.round(4)

        return cv_summary_df, cv_folds_df


    # ==============================
    # FIT PIPELINE (prep + clf)
    # ==============================
    @staticmethod
    def safe_name(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")


    @staticmethod
    def fit_pipeline(preprocessor, base_estimator, X_train, y_train, save_path=None):

        pipe = Pipeline([
            ("prep", clone(preprocessor)),
            ("clf",  clone(base_estimator)),
        ])
        pipe.fit(X_train, y_train)

        if save_path is not None:
            joblib.dump(pipe, save_path)

        return pipe

    # ==============================
    # SCORES CONTINUOS (PROBA / DECISION)
    # ==============================
    @staticmethod
    def get_scores(pipe, X_):

        clf = pipe
        if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
            clf = pipe.named_steps["clf"]

        if hasattr(pipe, "predict_proba") and callable(pipe.predict_proba):
            return pipe.predict_proba(X_)[:, 1]

        if hasattr(pipe, "decision_function") and callable(pipe.decision_function):
            s = pipe.decision_function(X_)
            s_min, s_max = np.min(s), np.max(s)
            return (s - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(s, dtype=float)

        return None

    # ==============================
    # MÉTRICAS (ENUNCIADO) + CM
    # ==============================
    @classmethod
    def compute_metrics_from_pipe(
        cls,
        pipe,
        X_eval,
        y_eval,
        threshold=None,
        scores_store=None,
        name=None
    ) -> dict:

        scores = cls.get_scores(pipe, X_eval)

        if (scores_store is not None) and (name is not None):
            scores_store[name] = scores

        auc = roc_auc_score(y_eval, scores) if scores is not None else np.nan

        if (threshold is not None) and (scores is not None):
            y_pred = (scores >= float(threshold)).astype(int)
        else:
            y_pred = pipe.predict(X_eval)

        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        return {
            "model": name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_eval, y_pred),
            "recall": recall_score(y_eval, y_pred, zero_division=0),
            "precision": precision_score(y_eval, y_pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(y_eval, y_pred, zero_division=0),
            "roc_auc": auc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    # ==============================
    # HOLDOUT (TRAIN -> EVAL)
    # ==============================
    @classmethod
    def evaluate_pipeline_holdout(
        cls,
        name,
        base_estimator,
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=None,
        scores_store=None,
        save_path=None
    ) -> dict:

        pipe = cls.fit_pipeline(preprocessor, base_estimator, X_train, y_train, save_path=save_path)

        metrics = cls.compute_metrics_from_pipe(
            pipe,
            X_test,
            y_test,
            threshold=threshold,
            scores_store=scores_store,
            name=name
        )

        return metrics


    @classmethod
    def run_holdout_and_persist(
        cls,
        MODEL_VARIANTS: dict,
        preprocessor,
        X_train, y_train,
        X_test, y_test,
        *,
        models_dir: str,
        subdir: str = "holdout",
        filename_prefix: str = "pipeline_",
        filename_suffix: str = "__holdout.pkl",
        show_progress: bool = True
    ):

        models_holdout_dir = os.path.join(models_dir, subdir)
        os.makedirs(models_holdout_dir, exist_ok=True)

        scores_store = {}
        rows = []

        items = MODEL_VARIANTS.items()
        if show_progress:
            items = tqdm(items, desc="Entrenando y guardando modelos holdout")

        for name, spec in items:
            save_path = os.path.join(
                models_holdout_dir,
                f"{filename_prefix}{cls.safe_name(name)}{filename_suffix}"
            )

            rows.append(
                cls.evaluate_pipeline_holdout(
                    name=name,
                    base_estimator=spec["estimator"],
                    preprocessor=preprocessor,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    threshold=spec.get("threshold", 0.50),
                    scores_store=scores_store,
                    save_path=save_path,
                )
            )

        results_df = pd.DataFrame(rows)

        return results_df, scores_store, models_holdout_dir


    # ==============================
    # EVALUACIÓN DE 1 FOLD (CV MANUAL)
    # ==============================
    @classmethod
    def eval_one_fold(cls, pipe, X_tr, y_tr, X_va, y_va, threshold=None) -> dict:

        pipe.fit(X_tr, y_tr)
        m = cls.compute_metrics_from_pipe(pipe, X_va, y_va, threshold=threshold, name=None)
        m.pop("model", None)
        return m


    # ==============================
    # GUARDADO FULL
    # ==============================
    @classmethod
    def run_full_train_and_persist(
        cls,
        MODEL_VARIANTS: dict,
        preprocessor,
        X,
        y,
        *,
        models_dir: str,
        results_holdout_df=None,
        models_to_persist=None,
        subdir: str = "full",
        filename_prefix: str = "pipeline_",
        filename_suffix: str = "__full.pkl"
    ):

        models_full_dir = os.path.join(models_dir, subdir)
        os.makedirs(models_full_dir, exist_ok=True)

        # --- Selección de modelos ---
        if models_to_persist is not None:
            selected_models = [m for m in models_to_persist if m in MODEL_VARIANTS]

        elif results_holdout_df is not None and "model" in results_holdout_df.columns:
            selected_models = [m for m in results_holdout_df["model"].tolist() if m in MODEL_VARIANTS]

        else:
            selected_models = list(MODEL_VARIANTS.keys())

        # --- Entrenamiento FULL + guardado ---
        saved_names = []

        for mname in selected_models:
            base_estimator = MODEL_VARIANTS[mname]["estimator"]

            fname = f"{filename_prefix}{cls.safe_name(mname)}{filename_suffix}"
            save_path = os.path.join(models_full_dir, fname)

            cls.fit_pipeline(
                preprocessor=preprocessor,
                base_estimator=base_estimator,
                X_train=X,
                y_train=y,
                save_path=save_path
            )

            saved_names.append(fname)

        # --- Tabla final (1 columna) ---
        saved_df = (
            pd.DataFrame({"pkl_full": saved_names})
              .sort_values("pkl_full")
              .reset_index(drop=True)
        )

        return saved_df, models_full_dir, saved_names, selected_models


    # ==============================
    # OBTENER THRESHOLD
    # ==============================
    @staticmethod
    def threshold_for_winner_pkl(cfg: dict, pkl_name: str, default: float = 0.50) -> float:

        # Sacar "logisticregression_max_recall" del nombre del pkl
        base = os.path.basename(pkl_name)
        m = re.match(r"^pipeline_(.+)__full\.pkl$", base)
        if not m:
            return float(default)

        safe_id = m.group(1).strip().lower()
        safe_id = safe_id.replace("__", "_")
        safe_id = re.sub(r"[^a-z0-9_]+", "_", safe_id)
        safe_id = re.sub(r"_+", "_", safe_id).strip("_")

        # Buscar en cfg["models"][model_key]["variants"][...]["threshold"]
        models_cfg = cfg.get("models", {})
        for model_key, spec in models_cfg.items():
            for v in (spec.get("variants") or []):
                variant_id = str(v.get("variant_id", "normal")).strip().lower()
                variant_id = re.sub(r"[^a-z0-9_]+", "_", variant_id)

                expected = f"{str(model_key).strip().lower()}_{variant_id}"
                expected = re.sub(r"[^a-z0-9_]+", "_", expected)
                expected = re.sub(r"_+", "_", expected).strip("_")

                if expected == safe_id:
                    try:
                        return float(v.get("threshold", default))
                    except Exception:
                        return float(default)

        return float(default)
