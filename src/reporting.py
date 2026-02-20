
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score


class ReportBuilder:

    # -----------------------------
    # TOP-N POR ESCENARIO
    # -----------------------------
    @staticmethod
    def objective_list(obj: Any) -> List[str]:

        if obj is None:
            return []
        if isinstance(obj, str):
            return [x.strip() for x in obj.split(",") if x.strip()]
        return list(obj)


    @staticmethod
    def parse_objective(objective: Any) -> Tuple[List[str], List[bool]]:

        cols: List[str] = []
        asc: List[bool] = []
        for item in ReportBuilder.objective_list(objective):
            if isinstance(item, str) and item.startswith("-"):
                cols.append(item[1:])
                asc.append(True)   # ASC
            else:
                cols.append(item)
                asc.append(False)  # DESC
        return cols, asc


    @staticmethod
    def get_top_df_by_scenario(results_df: pd.DataFrame, cfg: dict, sid: str) -> Tuple[pd.DataFrame, List[str]]:

        sc = cfg["scenarios"][sid]
        top_n = int(sc.get("top_n", 3))
        cols, asc = ReportBuilder.parse_objective(sc.get("objective", []))

        cols_ok = [c for c in cols if c in results_df.columns]
        if not cols_ok:
            raise ValueError(f"{sid}: objective no coincide con columnas del DataFrame")

        asc_ok = [asc[cols.index(c)] for c in cols_ok]

        df_top = (
            results_df
            .sort_values(by=cols_ok, ascending=asc_ok)
            .head(top_n)
            .reset_index(drop=True)
        )

        return df_top, cols_ok


    # -----------------------------
    # MATRIZ DE CONFUSIÓN (TP en [0,0])
    # -----------------------------
    @staticmethod
    def cm_tp00_from_counts(tn: int, fp: int, fn: int, tp: int) -> np.ndarray:

        return np.array([[int(tp), int(fp)], [int(fn), int(tn)]], dtype=int)


    @staticmethod
    def plot_confusion_from_counts(ax, tn: int, fp: int, fn: int, tp: int, title: str = "") -> None:

        cm = ReportBuilder.cm_tp00_from_counts(tn=tn, fp=fp, fn=fn, tp=tp)

        vmax = cm.max() if cm.size else 1
        im = ax.imshow(cm, vmin=0, vmax=vmax, cmap="Blues")

        # Texto con contraste automático
        thr_txt = vmax * 0.55
        for r in range(2):
            for c in range(2):
                val = int(cm[r, c])
                txt_color = "white" if val >= thr_txt else "black"
                ax.text(
                    c, r, str(val),
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=txt_color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.0, edgecolor="none")
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        # Etiquetas "Sí/No"
        ax.set_xticklabels(["Sí", "No"], fontsize=9)
        ax.set_yticklabels(["Sí", "No"], fontsize=9)

        ax.set_xlabel("Realidad", fontsize=9)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        ax.set_ylabel("Predicción", fontsize=9)
        ax.set_title(title, fontsize=9)


    # -----------------------------
    # ROC
    # -----------------------------
    @staticmethod
    def plot_roc_from_scores(
        ax,
        y_true,
        scores_store: Dict[str, np.ndarray],
        model_names: List[str],
    ) -> bool:

        any_curve = False

        for mname in model_names:
            scores = scores_store.get(mname, None) if scores_store is not None else None
            if scores is None:
                continue

            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, label=f"{mname} (AUC={auc:.3f})")
            any_curve = True

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title("ROC (HOLDOUT)", fontsize=9)
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.grid(True, alpha=0.25)

        if any_curve:
            ax.legend(fontsize=7, loc="lower right")

        return any_curve


    # -----------------------------
    # REPORTE COMPLETO POR ESCENARIO
    # -----------------------------
    @staticmethod
    def plot_scenario_report(
        sid: str,
        cfg: dict,
        results_df: pd.DataFrame,
        y_true,
        plt,
        scores_store: Optional[Dict[str, np.ndarray]] = None,
        set_name: str = "HOLDOUT",
        top_n_override: Optional[int] = None,
        show_table: bool = True,
        display_fn=None,
    ) -> Tuple[pd.DataFrame, Any]:

        df_top, objective_cols = ReportBuilder.get_top_df_by_scenario(results_df, cfg, sid)

        if top_n_override is not None:
            df_top = df_top.head(int(top_n_override)).reset_index(drop=True)

        # asegurar columna model
        if "model" not in df_top.columns and df_top.index.name == "model":
            df_top = df_top.reset_index()

        top_models = df_top["model"].tolist()
        n_models = len(top_models)

        print("\n" + "=" * 90)
        print(f"{sid} ({set_name}) — TOP {len(df_top)} por: {', '.join(objective_cols)}")
        print("=" * 90)

        if show_table:
            if display_fn is not None:
                display_fn(df_top)
            else:
                # fallback: print compacto
                print(df_top.to_string(index=False))

        FIGSIZE_BASE = 3.2
        DPI = 140

        fig, axes = plt.subplots(
            1, n_models + 1,
            figsize=(FIGSIZE_BASE * (n_models + 1), 2.9),
            dpi=DPI
        )

        # Si n_models=1, axes no es lista
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        # matrices
        for i, mname in enumerate(top_models):
            row = df_top[df_top["model"] == mname].iloc[0]
            tn, fp, fn, tp = int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"])
            ReportBuilder.plot_confusion_from_counts(
                axes[i],
                tn=tn, fp=fp, fn=fn, tp=tp,
                title=str(mname)
            )

        # roc
        ax_roc = axes[-1]
        if scores_store is not None:
            any_curve = ReportBuilder.plot_roc_from_scores(
                ax_roc, y_true=y_true, scores_store=scores_store, model_names=top_models
            )
            if not any_curve:
                ax_roc.text(0.5, 0.5, "Sin scores\npara ROC", ha="center", va="center", fontsize=9)
                ax_roc.set_axis_off()
        else:
            ax_roc.text(0.5, 0.5, "scores_store=None", ha="center", va="center", fontsize=9)
            ax_roc.set_axis_off()

        plt.tight_layout()
        plt.show()


    # ==============================
    # CORRELACIÓN (PLOT)
    # ==============================
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, plt, title: str = ""):
        num_df = df.select_dtypes(include="number").copy()

        if not num_df.empty:
            num_df = num_df.loc[:, num_df.nunique(dropna=True) > 1]

        if num_df.shape[1] == 0:
            print("⚠️ No hay columnas numéricas suficientes para calcular la correlación.")
            return pd.DataFrame()

        corr = num_df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        im = ax.imshow(corr.values, vmin=-1, vmax=1)

        if title == "":
            title = "Correlación (numéricas)"

        ax.set_title(title)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr.index, fontsize=7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()


    # ==============================
    # INGENIERÍA VARIABLES
    # ==============================
    @staticmethod
    def plot_attrition_rate_for_features(
        df: pd.DataFrame,
        *,
        target_col: str,
        target_positive_value: str,
        feature_cols: list,
        plt
    ):

        # Target binario y tasa base
        y = (df[target_col] == target_positive_value).astype(int)
        base_rate = y.mean()

        for col in feature_cols:
            if col not in df.columns:
                continue

            nunique = df[col].nunique(dropna=True)

            # -----------------------------
            # Binaria (0/1)
            # -----------------------------
            if nunique <= 2:
                g = (
                    pd.DataFrame({"group": df[col], "y": y})
                    .groupby("group", observed=True)["y"]
                    .agg(n="count", rate_yes="mean")
                    .reset_index()
                )
                g["rate_yes_pct"] = g["rate_yes"] * 100
                g = g.sort_values("group")

                x = g["group"].astype(str).tolist()
                y_pct = g["rate_yes_pct"].tolist()
                labels = [f"{p:.1f}% (n={n})" for p, n in zip(y_pct, g["n"].tolist())]

                fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=120)
                bars = ax.bar(x, y_pct)

                ax.set_title(f"Tasa de fuga por valor: {col}")
                ax.set_xlabel("Grupo")
                ax.set_ylabel(f"% {target_positive_value}")
                ax.tick_params(axis="x", labelrotation=25, labelsize=9)

                ax.axhline(base_rate * 100, linestyle="--")

                for bar, lab in zip(bars, labels):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        lab,
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

                plt.tight_layout()
                plt.show()

            # -----------------------------
            # Numérica -> 5 cuantiles
            # -----------------------------
            else:
                try:
                    bins = pd.qcut(df[col], q=5, duplicates="drop")
                except Exception:
                    # si no se puede binarizar (muchos iguales, etc.), lo saltamos sin romper
                    continue

                g = (
                    pd.DataFrame({"group": bins, "y": y})
                    .groupby("group", observed=True)["y"]
                    .agg(n="count", rate_yes="mean")
                    .reset_index()
                )
                g["rate_yes_pct"] = g["rate_yes"] * 100

                try:
                    g = g.sort_values("group")
                except Exception:
                    pass

                x = g["group"].astype(str).tolist()
                y_pct = g["rate_yes_pct"].tolist()
                labels = [f"{p:.1f}% (n={n})" for p, n in zip(y_pct, g["n"].tolist())]

                fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=120)
                bars = ax.bar(x, y_pct)

                ax.set_title(f"Tasa de fuga por bins: {col}")
                ax.set_xlabel("Bin (cuantiles)")
                ax.set_ylabel(f"% {target_positive_value}")
                ax.tick_params(axis="x", labelrotation=25, labelsize=8)

                ax.axhline(base_rate * 100, linestyle="--")

                for bar, lab in zip(bars, labels):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        lab,
                        ha="center",
                        va="bottom",
                        fontsize=7
                    )

                plt.tight_layout()
                plt.show()

        print(f"Tasa base ({target_col}={target_positive_value}): {base_rate*100:.2f}%")

        return base_rate
