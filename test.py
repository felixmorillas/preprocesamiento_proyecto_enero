
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_curve

from src.utils import load_yaml, load_dataset
from src.preprocessing import DataPreprocessor
from src.model_manager import ModelManager
from src.training import Training
from src.reporting import ReportBuilder

if __name__=="__main__":

    cfg = load_yaml('config.yml')

    # Columna objetivo
    TARGET              = cfg["target"]["col"]
    VALUE_TARGET        = cfg["target"]["value"]

    # Rutas
    DATASET_TEST_DIR    = cfg["paths"]['dataset_test']
    MODELS_FULL_DIR     = cfg["paths"]['models_full_dir']
    RESULTS_DIR         = cfg["paths"]["results_dir"]

    # Acciones sobre el dataset
    RUN_DROP_STATIC     = cfg["flags"]["run_drop_static"]
    RUN_TRANSFORMATION  = cfg["flags"]["run_transformation"]
    RUN_DROP_CORRELATED = cfg["flags"]["run_drop_correlated"]

    if not os.path.exists(DATASET_TEST_DIR):
        raise FileNotFoundError(f"No existe dataset_test: {DATASET_TEST_DIR}")

    test_df = load_dataset(DATASET_TEST_DIR)

    if TARGET not in test_df.columns:
        raise ValueError(f"El dataset ciego debe traer la columna target '{TARGET}' para poder calcular métricas.")

    # ==============================
    # PREPROCESADO MANUAL (MISMO QUE TRAIN)
    # ==============================
    if RUN_DROP_STATIC:

        DROP_USELESS_COLS = [
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ]

        test_df = DataPreprocessor.drop_columns(test_df, DROP_USELESS_COLS)

    if RUN_TRANSFORMATION:

        test_df = DataPreprocessor.add_feature_engineering(test_df)
    

    if RUN_DROP_CORRELATED:

        DROP_CORRELATED_COLS = [
            "EnvironmentSatisfaction",
            "JobLevel",
            "JobSatisfaction",
            "PerformanceRating",
            "RelationshipSatisfaction",
            "TotalWorkingYears",
            "WorkLifeBalance",
            "YearsInCurrentRole",
            "YearsWithCurrManager"
        ]

        test_df = DataPreprocessor.drop_columns(test_df, DROP_CORRELATED_COLS)

    # ==============================
    # X / y
    # ==============================
    y = (test_df[TARGET] == VALUE_TARGET).astype(int)
    X = test_df.drop(columns=[TARGET]).copy()

    # ==============================
    # GANADORES
    # ==============================
    winners = cfg.get("winners")
    if not winners:
        raise ValueError("\n⚠️ No hay 'winners' en el config o está vacío.")

    # opcional: por si luego quieres ROC
    scores_store = {}

    for scenario_key, pkl_name in sorted(winners.items()):

        model_path = os.path.join(MODELS_FULL_DIR, pkl_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No existe modelo ganador: {model_path}")

        # ==============================
        # CARGA PIPELINE YA ENTRENADO (FULL)
        # ==============================
        pipe = joblib.load(model_path)

        # threshold leído del bloque models/variants
        threshold = Training.threshold_for_winner_pkl(cfg, pkl_name, default=0.50)

        # ==============================
        # MÉTRICAS (sin reentrenar)
        # ==============================
        metrics = Training.compute_metrics_from_pipe(
            pipe,
            X,
            y,
            threshold=threshold,
            scores_store=scores_store,
            name=pkl_name
        )
                
        metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}

        model = metrics.get("model")
        thr   = metrics.get("threshold", None)

        acc  = metrics.get("accuracy")
        rec  = metrics.get("recall")
        prec = metrics.get("precision")
        spec = metrics.get("specificity")
        f1   = metrics.get("f1")
        auc  = metrics.get("roc_auc")

        tn = metrics.get("tn")
        fp = metrics.get("fp")
        fn = metrics.get("fn")
        tp = metrics.get("tp")

        # ==============================
        # PRINT BONITO
        # ==============================
        print("\n==============================")
        print(f"TEST CIEGO — {scenario_key.upper()} (FULL)")
        print("==============================")
        print(f"Modelo ganador:  {model}")
        print(f"Threshold usado: {float(thr):.2f}" if thr is not None and not (isinstance(thr, float) and np.isnan(thr))
              else "Threshold usado: (predict() del modelo)")

        print("\n--- Métricas ---")
        print(f"Accuracy:     {acc:.4f}")
        print(f"Recall:       {rec:.4f}")
        print(f"Precision:    {prec:.4f}")
        print(f"Specificity:  {spec:.4f}")
        print(f"F1:           {f1:.4f}")
        print(f"ROC-AUC:      {auc:.4f}" if not (auc is None or (isinstance(auc, float) and np.isnan(auc)))
              else "ROC-AUC:      (no disponible)")

        print("\n--- Matriz de confusión (tabla) ---")
        print("                 Real=Yes    Real=No")
        print(f"Pred=Yes (1)     {int(tp):7d}   {int(fp):7d}")
        print(f"Pred=No  (0)     {int(fn):7d}   {int(tn):7d}")


        # ==============================
        # GUARDAR IMAGEN DE LA CURVA ROC
        # ==============================
        scores = scores_store.get(pkl_name)  # porque en compute_metrics_from_pipe guardas scores_store[name] = scores

        if scores is None:
            print("ROC: no disponible (el modelo no devuelve scores continuos).")
        else:
            fpr, tpr, thr = roc_curve(y, scores)

            plt.figure(figsize=(5.6, 4.2), dpi=140)
            plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Azar")
            plt.xlabel("FPR (1 - Specificity)")
            plt.ylabel("TPR (Recall)")
            plt.title(f"ROC — {scenario_key.upper()} — {model}")
            plt.legend(loc="lower right")
            plt.tight_layout()

            os.makedirs(RESULTS_DIR, exist_ok=True)
            out_path = os.path.join(RESULTS_DIR, f"roc_{scenario_key.lower()}_{pkl_name.replace('.pkl','')}.png")
            plt.savefig(out_path)
            plt.close()


        # ==============================
        # GUARDAR MATRIZ DE CONFUSIÓN
        # ==============================
        fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)

        ReportBuilder.plot_confusion_from_counts(
            ax=ax,
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            title=f"CM — {scenario_key.upper()} — {pkl_name}"
        )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(
            RESULTS_DIR,
            f"cm_{scenario_key.lower()}_{pkl_name.replace('.pkl','')}.png"
        )

        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        print("\n------------------------------------------------")
