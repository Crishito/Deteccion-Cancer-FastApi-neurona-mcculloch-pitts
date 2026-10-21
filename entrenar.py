"""Entrenamiento académico de una neurona McCulloch-Pitts para WDBC."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_mp.pkl"
NUM_CARACTERISTICAS = 8


class MpNeurona:
    """Clasificador M-P sin pesos: predice 1 si la suma supera el umbral."""

    def __init__(self):
        self.threshold = None

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Entrena primero la neurona.")
        matriz = np.asarray(X, dtype=int)
        return (matriz.sum(axis=1) >= self.threshold).astype(int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int)
        mejores = None
        for umbral in range(X.shape[1] + 1):
            self.threshold = umbral
            pred = self.predict(X)
            exactitud = accuracy_score(y, pred)
            # Ante igual exactitud, preferir sensibilidad para la clase maligna (0).
            sensibilidad_malignos = recall_score(y, pred, pos_label=0, zero_division=0)
            puntaje = (exactitud, sensibilidad_malignos)
            if mejores is None or puntaje > mejores:
                mejores = puntaje
                mejor_umbral = umbral
        self.threshold = mejor_umbral
        return self


def main():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target  # 0: maligno; 1: benigno

    X_ent, X_prueba, y_ent, y_prueba = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Seleccionar SOLO con entrenamiento: 8 atributos más asociados con la clase
    # según prueba ANOVA F (no equivale a validación clínica).
    selector = SelectKBest(score_func=f_classif, k=NUM_CARACTERISTICAS)
    X_ent_8 = selector.fit_transform(X_ent, y_ent)
    X_prueba_8 = selector.transform(X_prueba)
    nombres = dataset.feature_names[selector.get_support()].tolist()

    # Los límites se aprenden ÚNICAMENTE del entrenamiento y luego se guardan.
    cortes = np.median(X_ent_8, axis=0)
    # Conservar convención del cuaderno: valor bajo=1, valor alto=0.
    X_ent_bin = (X_ent_8 <= cortes).astype(int)
    X_prueba_bin = (X_prueba_8 <= cortes).astype(int)

    neurona = MpNeurona().fit(X_ent_bin, y_ent)
    pred = neurona.predict(X_prueba_bin)

    artefacto = {
        "tipo": "Neurona McCulloch-Pitts (suma y umbral)",
        "caracteristicas": nombres,
        "cortes": cortes.tolist(),
        "threshold": int(neurona.threshold),
        "clases": {0: "Maligno", 1: "Benigno"},
        "convencion": "1 si valor <= mediana de entrenamiento; 0 en otro caso",
    }
    joblib.dump(artefacto, RUTA_MODELO)

    print("Modelo guardado en:", RUTA_MODELO)
    print("Características seleccionadas:")
    for nombre, corte in zip(nombres, cortes):
        print(f"  - {nombre}: corte = {corte:.5f}")
    print("Umbral M-P:", neurona.threshold)
    print(f"Exactitud de prueba: {accuracy_score(y_prueba, pred):.2%}")
    print(f"Sensibilidad a malignos: {recall_score(y_prueba, pred, pos_label=0):.2%}")
    print("Matriz de confusión [filas reales 0,1; columnas predichas 0,1]:")
    print(confusion_matrix(y_prueba, pred, labels=[0, 1]))
    print("AVISO: prototipo educativo, no apto para diagnóstico médico.")


if __name__ == "__main__":
    main()
