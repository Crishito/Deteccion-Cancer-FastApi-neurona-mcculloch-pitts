import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class MpNeurona:
    def __init__(self):
        self.threshold = None
    def model(self, x):
        z = sum(x)
        return (z >= self.threshold)
    def predict(self, X):
        return np.array([self.model(x) for x in X])
    def fit(self, X, Y):
        from sklearn.metrics import accuracy_score
        accuracy = {}
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        self.threshold = max(accuracy, key=accuracy.get)

# 1. Carga y partición
cancer_mama = load_breast_cancer()
df = pd.DataFrame(cancer_mama.data, columns=cancer_mama.feature_names)
X_train, X_test, y_train, y_test = train_test_split(df, cancer_mama.target, stratify=cancer_mama.target, random_state=42)

# 2. Binarización
X_train_bin = X_train.apply(pd.cut, bins=2, labels=[1,0])

# 3. Filtrar las 8 características determinantes
caracteristicas = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean smoothness', 
    'mean compactness', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
]
X_train_reducido = X_train_bin[caracteristicas]

# 4. Entrenar y guardar
neurona = MpNeurona()
neurona.fit(X_train_reducido.to_numpy(), y_train)
joblib.dump(neurona, 'modelo_mp.pkl')
print(f"Modelo exportado exitosamente. Threshold óptimo: {neurona.threshold}")