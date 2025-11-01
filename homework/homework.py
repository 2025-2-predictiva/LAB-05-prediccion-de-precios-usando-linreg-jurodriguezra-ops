#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import gzip
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURACIÓN Y PARÁMETROS
# ============================================================
class Setup:
    """Contenedor de configuraciones globales (no modificar rutas)."""
    base_dir = Path(__file__).resolve().parent.parent
    files = {
        "train_zip": base_dir / "files" / "input" / "train_data.csv.zip",
        "test_zip": base_dir / "files" / "input" / "test_data.csv.zip",
        "train_name": "train_data.csv",
        "test_name": "test_data.csv",
        "model_out": base_dir / "files" / "models" / "model.pkl.gz",
        "metrics_out": base_dir / "files" / "output" / "metrics.json"
    }

    # Columnas y parámetros del modelo
    target = "Present_Price"
    year_ref = 2021
    cat_vars = ["Fuel_Type", "Selling_type", "Transmission"]
    num_vars = ["Driven_kms", "Owner", "Age"]
    passthrough = ["Selling_Price"]

    grid_params = {"select__k": list(range(4, 12))}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def read_zip_csv(zip_path: Path, csv_name: str) -> pd.DataFrame:
    """Extrae y carga un CSV desde un archivo ZIP."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(csv_name) as f:
            df = pd.read_csv(f)
    # eliminar índice fantasma si existe
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos antes del modelado."""
    df = df.copy()
    df["Age"] = Setup.year_ref - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)
    df.dropna(inplace=True)
    df["Selling_Price"] = np.log1p(df["Selling_Price"])
    return df


# ============================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================
def assemble_model() -> GridSearchCV:
    """Crea pipeline con preprocesamiento, selección y regresión."""

    prep = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), Setup.cat_vars),
            ("numeric", MinMaxScaler(), Setup.num_vars),
            ("passthrough", "passthrough", Setup.passthrough),
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("prep", prep),
        ("select", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression())
    ])

    tuned = GridSearchCV(
        estimator=model,
        param_grid=Setup.grid_params,
        cv=10,
        scoring="neg_mean_absolute_error",
        refit=True,
        verbose=1
    )

    return tuned


# ============================================================
# MÉTRICAS Y GUARDADO
# ============================================================
def compute_scores(tag: str, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    """Calcula y devuelve métricas principales."""
    return {
        "type": "metrics",
        "dataset": tag,
        "r2": float(r2_score(actual, predicted)),
        "mse": float(mean_squared_error(actual, predicted)),
        "mad": float(mean_absolute_error(actual, predicted)),
    }


def export_model(obj: Any, destination: Path) -> None:
    """Guarda el modelo entrenado en formato comprimido .pkl.gz."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wb") as file:
        pickle.dump(obj, file)


def export_metrics(metrics: List[Dict[str, Any]], destination: Path) -> None:
    """Escribe las métricas obtenidas en formato JSONL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")


# ============================================================
# PROCESO PRINCIPAL
# ============================================================
def execute_pipeline():
    """Ejecuta todo el flujo de entrenamiento y evaluación."""

    # --- Cargar y preparar ---
    train_df = transform_data(read_zip_csv(Setup.files["train_zip"], Setup.files["train_name"]))
    test_df = transform_data(read_zip_csv(Setup.files["test_zip"], Setup.files["test_name"]))

    X_train = train_df.drop(columns=[Setup.target])
    y_train = np.log1p(train_df[Setup.target])
    X_test = test_df.drop(columns=[Setup.target])
    y_test = np.log1p(test_df[Setup.target])

    # --- Entrenamiento ---
    searcher = assemble_model()
    searcher.fit(X_train, y_train)

    # --- Guardar modelo ---
    export_model(searcher, Setup.files["model_out"])

    # --- Predicciones y métricas ---
    train_preds = np.expm1(searcher.predict(X_train))
    test_preds = np.expm1(searcher.predict(X_test))
    train_real = np.expm1(y_train)
    test_real = np.expm1(y_test)

    metrics_list = [
        compute_scores("train", train_real, train_preds),
        compute_scores("test", test_real, test_preds)
    ]

    # --- Guardar métricas ---
    export_metrics(metrics_list, Setup.files["metrics_out"])

    print("✅ Proceso completado: modelo y métricas exportados correctamente.")


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    execute_pipeline()
