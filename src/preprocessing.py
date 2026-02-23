
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:

    @staticmethod
    def drop_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:

        return df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()


    @staticmethod
    def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

        df_out = df.copy()

        required_cols = [
            "BusinessTravel",
            "Department",
            "JobLevel",
            "MonthlyIncome",
            "OverTime",
            "WorkLifeBalance",
            "YearsAtCompany",
            "YearsInCurrentRole"
        ]

        missing = [c for c in required_cols if c not in df_out.columns]
        if missing:
            raise KeyError(f"Faltan columnas necesarias para feature engineering: {missing}")

        df_out["new_hire_flag"] = (df_out["YearsAtCompany"] <= 1).astype(int)

        df_out["overtime_and_travel"] = (
            (df_out["OverTime"] == "Yes") & (df_out["BusinessTravel"] == "Travel_Frequently")
        ).astype(int)

        df_out["sales_and_overtime"] = (
            (df_out["Department"] == "Sales") & (df_out["OverTime"] == "Yes")
        ).astype(int)

        df_out["role_stability_ratio"] = df_out["YearsInCurrentRole"] / (df_out["YearsAtCompany"] + 1)

        df_out["income_per_joblevel"] = df_out["MonthlyIncome"] / (df_out["JobLevel"] + 1)

        # Compuesto de satisfacción
        # axis=1 es para que calcule la operación por fila, no por columna
        sat_cols = [
            "EnvironmentSatisfaction",
            "JobSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance"
        ]
        df_out["satisfaction_mean"] = df_out[sat_cols].mean(axis=1)

        return df_out


    @staticmethod
    def build_preprocessor_ohe_from_X(X: pd.DataFrame, scaler: str="StandardScaler", encoder: str="OneHotEncoder"):

        # ------------------------------
        # Columnas numéricas y categóricas
        # ------------------------------
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # ------------------------------
        # Scaler configurable
        # ------------------------------
        scaler = (scaler or "StandardScaler").strip()

        if scaler == "StandardScaler":
            scaler_obj = StandardScaler()
        elif scaler == "RobustScaler":
            scaler_obj = RobustScaler()
        elif scaler.lower() in {"none", "no"}:
            scaler_obj = None
        else:
            raise ValueError(f"Scaler no soportado: {scaler}")

        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scaler_obj is not None:
            num_steps.append(("scaler", scaler_obj))

        num_pipe = Pipeline(steps=num_steps)

        # ------------------------------
        # Encoder configurable
        # ------------------------------
        encoder = (encoder or "OneHotEncoder").strip()

        if encoder == "OneHotEncoder":
            encoder_obj = OneHotEncoder(
                handle_unknown="ignore",
                drop="if_binary",
                sparse_output=False
            )
            cat_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder_obj),
            ])

        elif encoder == "OrdinalEncoder":
            # Para features no se usa LabelEncoder (es para y).
            # La alternativa correcta en sklearn es OrdinalEncoder.
            encoder_obj = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            cat_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder_obj),
            ])

        elif encoder.lower() in {"none", "no"}:
            cat_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ])

        else:
            raise ValueError(f"Encoder no soportado: {encoder}")

        # ------------------------------
        # ColumnTransformer final
        # ------------------------------
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        return preprocessor, num_cols, cat_cols
