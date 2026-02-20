
import numpy as np
import pandas as pd


class EDAHelper:

    def __init__(self, target: str, max_examples: int, sep_len: int):

        self.target        = target
        self.sep_len       = sep_len
        self.max_examples  = max_examples
        self.empty_markers = {"", "na", "n/a", "null", "none", "-", "--"}

        # Diccionario de significado de columnas
        self.significado = {
            "Age": "Edad del empleado (18-60).",
            "Attrition": "Indica si el empleado dejó la empresa (Yes/No).",
            "BusinessTravel": "Frecuencia de viajes: Non-Travel, Travel_Rarely, Travel_Frequently",
            "DailyRate": "Tasa salarial diaria (valor numérico sin significado especial).",
            "Department": "Departamento: Sales, Research & Development, Human Resources.",
            "DistanceFromHome": "Distancia en millas desde el hogar al trabajo.",
            "Education": "Nivel educativo (1-5).",
            "EducationField": "Campo educativo: Life Sciences, Medical, Marketing, Technical Degree, etc.",
            "EmployeeCount": "Constante = 1 (sin utilidad analítica).",
            "EnvironmentSatisfaction": "Satisfacción con el entorno laboral (1-4).",
            "Gender": "Género del empleado: Male/Female.",
            "HourlyRate": "Tasa salarial por hora (valor numérico aleatorio).",
            "JobInvolvement": "Nivel de implicación en el trabajo (1-4).",
            "JobLevel": "Nivel del puesto (1-5).",
            "JobRole": "Rol laboral: Sales Executive, Research Scientist, etc.",
            "JobSatisfaction": "Satisfacción con el puesto (1-4).",
            "MaritalStatus": "Estado civil: Single, Married, Divorced.",
            "MonthlyIncome": "Ingresos mensuales del empleado.",
            "MonthlyRate": "Tasa salarial mensual (numérico sin significado real).",
            "NumCompaniesWorked": "Número de empresas previas en las que trabajó.",
            "Over18": "Constante = Yes.",
            "OverTime": "Si trabaja horas extra: Yes/No.",
            "PercentSalaryHike": "Porcentaje de incremento salarial anual.",
            "PerformanceRating": "Valoración del desempeño (1-4, normalmente 3-4).",
            "RelationshipSatisfaction": "Satisfacción con relaciones laborales (1-4).",
            "StandardHours": "Horas estándar = 80 (constante).",
            "StockOptionLevel": "Nivel de opciones sobre acciones (0-3).",
            "TotalWorkingYears": "Años totales trabajados en la carrera profesional.",
            "TrainingTimesLastYear": "Número de formaciones completadas el último año.",
            "WorkLifeBalance": "Equilibrio vida-trabajo (1-4).",
            "YearsAtCompany": "Años trabajados en la empresa actual.",
            "YearsInCurrentRole": "Años en el puesto actual.",
            "YearsSinceLastPromotion": "Años desde la última promoción.",
            "YearsWithCurrManager": "Años con el manager actual.",
        }


    # ----------------------------
    # UTILIDADES DE IMPRESIÓN
    # ----------------------------
    @staticmethod
    def print_header(title: str) -> None:

        print("\n" + "=" * 80)
        print(f"📌 {title}")
        print("=" * 80 + "\n")


    # ----------------------------
    # CHECKS BÁSICOS
    # ----------------------------
    def get_dimensions(self, df: pd.DataFrame, title: str) -> None:

        self.print_header(title)
        print(f"  - Filas: {df.shape[0]:,}")
        print(f"  - Columnas: {df.shape[1]:,}")


    def check_null_values(self, df: pd.DataFrame, title: str) -> None:

        self.print_header(title)
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if null_cols.empty:
            print("  - No hay valores nulos")
        else:
            print("  - Sí hay valores nulos")
            print(null_cols)


    def check_duplicates(self, df: pd.DataFrame, title: str) -> None:

        self.print_header(title)
        print(f"  - Filas duplicadas: {df.duplicated().sum()}")


    def columns_with_blanks(self, df: pd.DataFrame, title: str) -> None:

        self.print_header(title)

        total_blanks = 0
        for col in df.columns:
            s = df[col]

            is_text_like = (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)   # incluye string[python]
                or pd.api.types.is_categorical_dtype(s)
            )

            if not is_text_like:
                continue

            blanks_col = (s.astype("string").str.strip() == "").sum()

            if blanks_col > 0:
                print(f"⚠ {col}: {int(blanks_col)} valores vacíos")
                total_blanks += int(blanks_col)

        if total_blanks == 0:
            print("  - No hay valores vacíos")

        print("  ✔ Comprobación completa.")

    # ----------------------------
    # DETECTORES (para limpieza smart)
    # ----------------------------
    def has_blank_strings(self, df: pd.DataFrame) -> bool:
        for col in df.columns:
            s = df[col]
            is_text_like = (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_categorical_dtype(s)
            )
            if not is_text_like:
                continue
            if (s.astype("string").str.strip() == "").any():
                return True
        return False

    @staticmethod
    def has_nulls(df: pd.DataFrame) -> bool:
        return bool(df.isnull().any().any())

    # ----------------------------
    # RESÚMENES
    # ----------------------------
    def unique_values_by_column(self, df: pd.DataFrame, title: str) -> None:

        self.print_header(title)

        constant_columns = []

        prefix_width = max(len(f"  - {col}:") for col in df.columns)
        max_unique_count = max(df[col].nunique(dropna=False) for col in df.columns)
        count_width = len(str(max_unique_count))

        const_prefix_width = max(len(f"  - {col}:") for col in df.columns)

        for col in df.columns:
            unique_values = df[col].drop_duplicates().to_numpy()
            unique_count = len(unique_values)

            if unique_count == 1:
                constant_columns.append((col, unique_values[0]))
                continue

            examples = unique_values[: self.max_examples]
            if df[col].dtype == "object":
                examples_str = "[" + " ".join(repr(x) for x in examples) + "]"
            else:
                examples_str = "[" + " ".join(str(x) for x in examples) + "]"

            prefix = f"  - {col}:"
            pad = " " * (prefix_width - len(prefix) + 1)
            print(f"{prefix}{pad}{str(unique_count).rjust(count_width)} valores: {examples_str}")
            print("  " + "-" * self.sep_len)

        if constant_columns:
            print("\n" + "=" * self.sep_len)
            print("COLUMNAS CON UN ÚNICO VALOR (CONSTANTES)")
            print("=" * self.sep_len)

            for col, value in sorted(constant_columns, key=lambda x: x[0].lower()):
                value_str = repr(value) if isinstance(value, str) else str(value)
                prefix = f"  - {col}:"
                pad = " " * (const_prefix_width - len(prefix) + 1)
                print(f"{prefix}{pad}{value_str}")
                print("  " + "-" * self.sep_len)


    def numeric_and_categoric_distribution(self, df: pd.DataFrame, title: str) -> None:

        feature_cols = [c for c in df.columns if c != self.target]
        num_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        self.print_header(title)
        print(f"  - Numéricas:   {len(num_cols)}")
        print(f"  - Categóricas: {len(cat_cols)}")
        print(f"  - Target:      {self.target} (positivo = Yes)\n")


    def data_type_by_columns(self, df: pd.DataFrame, title: str) -> None:

        self.print_header(title)

        grupos: dict[str, list[str]] = {}
        for col, dt in df.dtypes.items():
            grupos.setdefault(str(dt), []).append(col)

        max_len = max(len(c) for c in df.columns)

        for dt in sorted(grupos.keys()):
            print(f"- {dt}:")
            print("  " + "-" * self.sep_len)

            for col in sorted(grupos[dt]):
                name = f"🎯 {col}" if col == self.target else col
                desc = self.significado.get(col, "(sin descripción)")
                gap = "" if col == self.target else " "
                print(f"  - {name:<{max_len}}{gap}{desc}")
                print("  " + "-" * self.sep_len)

            print()

    # ----------------------------
    # TARGET
    # ----------------------------
    def target_distribution(self, df: pd.DataFrame, plt, title: str | None = None) -> None:

        target = self.target
        counts = df[target].value_counts().reindex(["No", "Yes"]).fillna(0).astype(int)
        perc = (counts / counts.sum() * 100).round(2)

        x = counts.index.tolist()
        y = counts.values.tolist()
        labels = [f"{c} ({p:.2f}%)" for c, p in zip(counts.values, perc.values)]

        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=120)
        bars = ax.bar(x, y)

        ax.set_title(title or f"Distribución de la variable objetivo ({target})")
        ax.set_xlabel(target)
        ax.set_ylabel("Filas")

        for bar, lab in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                lab,
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_ylim(0, max(y) * 1.15 if max(y) > 0 else 1)
        plt.tight_layout()
        plt.show()


    # ----------------------------
    # TRATAMIENTOS
    # ----------------------------
    @staticmethod
    def apply_null_values(df: pd.DataFrame) -> pd.DataFrame:

        df_out = df.copy()

        num_cols = df_out.select_dtypes(include=["number"]).columns
        cat_cols = df_out.select_dtypes(exclude=["number"]).columns

        num_cols_with_nulls = [c for c in num_cols if df_out[c].isnull().any()]
        cat_cols_with_nulls = [c for c in cat_cols if df_out[c].isnull().any()]

        if num_cols_with_nulls:
            for c in num_cols_with_nulls:
                df_out[c] = df_out[c].fillna(df_out[c].median())
            print(f"  - Numéricas imputadas con mediana: {len(num_cols_with_nulls)}")

        if cat_cols_with_nulls:
            for c in cat_cols_with_nulls:
                mode = df_out[c].mode(dropna=True)
                df_out[c] = df_out[c].fillna(mode.iloc[0] if not mode.empty else "Missing")
            print(f"  - Categóricas imputadas con moda: {len(cat_cols_with_nulls)}")

        return df_out

    def fill_empty_only(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        for c in df.columns:
            s = df[c].astype("string").str.strip()
            m = s.isna() | (s == "") | s.str.lower().isin(self.empty_markers)

            if not m.any():
                continue

            df.loc[m, c] = np.nan

            if pd.api.types.is_numeric_dtype(df[c]):
                df.loc[m, c] = df[c].median()
            else:
                mode = df[c].mode(dropna=True)
                df.loc[m, c] = (mode.iloc[0] if not mode.empty else "Missing")

        return df

    def clean_missing(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:

        df_out = df.copy()

        if self.has_blank_strings(df_out):
            if verbose:
                self.columns_with_blanks(df_out, "Detectado → vacíos en texto")
            df_out = self.fill_empty_only(df_out)

        if self.has_nulls(df_out):
            if verbose:
                self.check_null_values(df_out, "Detectado → nulos (NaN)")
            df_out = self.apply_null_values(df_out)

        return df_out

    # ----------------------------
    # PIPE DE "DETALLES DATASET"
    # ----------------------------
    def show_dataset_details(self, df: pd.DataFrame, plt=None) -> None:

        self.get_dimensions(df, "Dimensiones del dataset")

        self.check_null_values(df, "Valores nulos por columna")

        self.check_duplicates(df, "Duplicados")

        self.columns_with_blanks(df, "Columnas con strings vacíos o espacios en blanco")

        self.unique_values_by_column(df, f"Valores únicos por columna (primeros {self.max_examples})")

        self.numeric_and_categoric_distribution(df, "Distribución tipos de datos")

        self.data_type_by_columns(df, "Tipos de datos por columna")

        if plt is not None:
            self.target_distribution(df, plt)


    # ----------------------------
    # OUTLIERS
    # ----------------------------
    @staticmethod
    def iqr_outlier_summary(df: pd.DataFrame, target: str, drop_cols: list) -> pd.DataFrame:

        df2 = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

        # target binaria
        y = (df2[target] == "Yes").astype(int)
        Xnum = df2.select_dtypes(include=[np.number]).copy()

        rows = []
        for col in Xnum.columns:
            s = Xnum[col].dropna()
            if s.empty:
                continue

            vmin = s.min()
            vmax = s.max()

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            # máscara outlier sobre el df completo (respeta NaN)
            mask = (df2[col] < lower) | (df2[col] > upper)

            n_total = df2[col].notna().sum()
            n_out = int(mask.sum())
            pct_out = (n_out / n_total * 100) if n_total else 0.0

            # por clase
            mask_yes = mask & (y == 1)
            mask_no  = mask & (y == 0)

            n_yes = int((y == 1).sum())
            n_no  = int((y == 0).sum())

            out_yes = int(mask_yes.sum())
            out_no  = int(mask_no.sum())

            pct_yes = (out_yes / n_yes * 100) if n_yes else 0.0
            pct_no  = (out_no / n_no * 100) if n_no else 0.0

            rows.append({
                "variable": col,
                "n_non_null": int(n_total),
                "min_real": float(vmin),
                "max_real": float(vmax),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower": float(lower),
                "upper": float(upper),
                "outliers_count": n_out,
                "outliers_pct": pct_out,
                "outliers_yes": out_yes,
                "outliers_no": out_no,
                "outliers_yes_pct_in_class": pct_yes,
                "outliers_no_pct_in_class": pct_no,
            })

        res = pd.DataFrame(rows)

        float_cols = [
            "min_real","max_real","q1","q3","iqr","lower","upper","outliers_pct",
            "outliers_yes_pct_in_class","outliers_no_pct_in_class"
        ]
        for c in float_cols:
            if c in res.columns:
                res[c] = res[c].round(2)

        # ordena por % outliers (desc)
        res = res.sort_values(["outliers_pct", "outliers_count"], ascending=False).reset_index(drop=True)

        return res


    @staticmethod
    def plot_outlier_pct_bar(summary: pd.DataFrame, top_n: int, plt):

        s = summary.head(top_n).copy()

        x = s["variable"].astype(str).tolist()
        y = s["outliers_pct"].astype(float).tolist()
        labels = [f"{v:.2f}%" for v in y]

        fig, ax = plt.subplots(figsize=(max(6, top_n * 0.6), 3.8), dpi=120)
        bars = ax.bar(x, y)

        ax.set_title(f"Top {top_n} variables con mayor % de outliers (IQR)")
        ax.set_xlabel("Variable")
        ax.set_ylabel("Outliers (%)")

        # Rotación de ticks
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)

        # Etiquetas encima de cada barra
        for bar, lab in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                lab,
                ha="center",
                va="bottom",
                fontsize=8
            )

        ax.set_ylim(0, max(y) * 1.15 if len(y) and max(y) > 0 else 1)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_boxplots_top_outliers(df: pd.DataFrame, summary: pd.DataFrame, target: str, top_n: int, plt):

        top_vars = summary.head(top_n)["variable"].tolist()

        # Orden fijo del target si es Attrition típico
        if target in df.columns:
            order = [v for v in ["No", "Yes"] if v in df[target].dropna().unique().tolist()]
            if not order:
                order = sorted(df[target].dropna().unique().tolist())
        else:
            raise KeyError(f"No existe la columna target '{target}' en df.")

        for col in top_vars:
            if col not in df.columns:
                continue

            tmp = df[[col, target]].dropna()
            if tmp.empty:
                continue

            data = []
            labels = []
            for grp in order:
                vals = tmp.loc[tmp[target] == grp, col].dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    labels.append(grp)

            if not data:
                continue

            fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=120)

            ax.boxplot(
                data,
                labels=labels,
                showfliers=True, # muestra outliers
                patch_artist=False
            )

            ax.set_title(f"Boxplot por {target} - {col} (top outliers)")
            ax.set_xlabel(target)
            ax.set_ylabel(col)

            plt.tight_layout()
            plt.show()
