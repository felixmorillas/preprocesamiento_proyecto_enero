
import re
import joblib

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ModelManager:

    # -----------------------------
    # Map: nombre en YAML -> clase sklearn
    # -----------------------------
    MODEL_CLASS_MAP = {
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GaussianNB": GaussianNB,
        "LogisticRegression": LogisticRegression,
        "SVC": SVC,
    }

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}


    # ==============================
    # HELPERS
    # ==============================
    @staticmethod
    def _safe_tag(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

    def _resolve_params_placeholders(self, params: dict) -> dict:

        if not params:
            return {}

        out = dict(params)
        rs = None
        try:
            rs = self.cfg["split"]["random_state"]
        except Exception:
            # Si no existe, no rompe; simplemente no resuelve placeholders
            rs = None

        for k, v in out.items():
            if isinstance(v, str) and v.strip() == "${split.random_state}" and rs is not None:
                out[k] = rs

        return out

    # ==============================
    # VARIANTES DE MODELOS (DESDE CONFIG)
    # ==============================
    def build_model_variants_from_config(self, cfg: dict | None = None) -> dict:


        if cfg is not None:
            self.cfg = cfg or {}

        if not self.cfg:
            raise ValueError("No hay cfg cargada. Usa set_config(cfg) o pasa cfg al método.")

        out = {}

        models_cfg = self.cfg.get("models", {})
        if not isinstance(models_cfg, dict) or not models_cfg:
            raise ValueError("cfg['models'] no existe o está vacío.")

        for model_key, spec in models_cfg.items():
            if "class" not in spec:
                raise ValueError(f"Falta 'class' en cfg['models']['{model_key}'].")

            class_name = spec["class"]
            if class_name not in self.MODEL_CLASS_MAP:
                raise ValueError(
                    f"Clase no soportada en config: {class_name} (modelo: {model_key})"
                )

            # Nuevo formato (variants) o fallback al viejo (params)
            variants = spec.get("variants")
            if not variants:
                variants = [{
                    "variant_id": "normal",
                    "threshold": 0.50,
                    "params": spec.get("params", {}) or {}
                }]

            for v in variants:
                variant_id = v.get("variant_id", "normal")
                threshold = float(v.get("threshold", 0.50))
                params = self._resolve_params_placeholders(v.get("params", {}) or {})

                estimator_cls = self.MODEL_CLASS_MAP[class_name]
                estimator = estimator_cls(**params)

                full_id = f"{self._safe_tag(model_key)}__{self._safe_tag(variant_id)}"

                out[full_id] = {
                    "id": full_id,
                    "model_key": model_key,
                    "variant_id": variant_id,
                    "class_name": class_name,
                    "params": params,
                    "threshold": threshold,
                    "estimator": estimator,
                }

        return out


    # ==============================
    # UTILIDADES DE INSPECCIÓN
    # ==============================
    @staticmethod
    def safe_name(name: str) -> str:
        s = str(name).strip().lower().replace("__", "_")
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    @classmethod
    def select_winners_by_scenarios(
        cls,
        cfg: dict,
        results_holdout_df,
        *,
        scenario_map=None,
        full_prefix: str = "pipeline_",
        full_suffix: str = "__full.pkl"
    ):

        if scenario_map is None:
            scenario_map = [("S1", "scenario_1"), ("S2", "scenario_2"), ("S3", "scenario_3")]

        if "scenarios" not in cfg:
            raise KeyError("cfg no contiene la clave 'scenarios'.")

        if results_holdout_df is None or len(results_holdout_df) == 0:
            raise ValueError("results_holdout_df está vacío o es None.")

        winners = {}

        for sid, key in scenario_map:
            if sid not in cfg["scenarios"]:
                raise KeyError(f"No existe cfg['scenarios']['{sid}'].")

            obj = cfg["scenarios"][sid].get("objective")
            if obj is None:
                raise KeyError(f"No existe 'objective' en cfg['scenarios']['{sid}'].")

            # Normalizar objective -> lista de columnas
            cols = [
                t.strip()
                for t in (obj if isinstance(obj, (list, tuple)) else str(obj).split(","))
                if t.strip()
            ]

            # Validar columnas
            missing = [c for c in cols if c not in results_holdout_df.columns]
            if missing:
                raise KeyError(
                    f"En escenario {sid}, faltan columnas en RESULTS_HOLDOUT_DF: {missing}"
                )

            # Elegir ganador (asume "más alto es mejor" en todas)
            winner_model = (
                results_holdout_df
                .sort_values(cols, ascending=[False] * len(cols))
                .iloc[0]["model"]
            )

            s = cls.safe_name(winner_model)
            winners[key] = f"{full_prefix}{s}{full_suffix}"

        # Bloque YAML listo para pegar
        lines = ["winners:"]
        for k, v in winners.items():
            lines.append(f"  {k}: {v}")
        yaml_block = "\n".join(lines)

        return winners, yaml_block
