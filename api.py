from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import __main__  # <-- NUEVO: Importamos el módulo principal

# Instanciamos la clase de la neurona
class MpNeurona:
    def __init__(self):
        self.threshold = None
    def model(self, x):
        return sum(x) >= self.threshold
    def predict(self, X):
        return np.array([self.model(x) for x in X])

# <-- SOLUCIONAR EL ERROR DE JOBLIB -->
__main__.MpNeurona = MpNeurona 

app = FastAPI()

# Permisos para que la página HTML pueda comunicarse con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos el modelo (ahora sí funcionará)
modelo = joblib.load('modelo_mp.pkl')

# Definimos el esquema de las 8 características determinantes
class DatosPaciente(BaseModel):
    mean_radius: int
    mean_texture: int
    mean_perimeter: int
    mean_smoothness: int
    mean_compactness: int
    mean_concave_points: int
    mean_symmetry: int
    mean_fractal_dimension: int

@app.post("/diagnostico")
def predecir(datos: DatosPaciente):
    entrada = np.array([[
        datos.mean_radius, datos.mean_texture, datos.mean_perimeter, 
        datos.mean_smoothness, datos.mean_compactness, datos.mean_concave_points, 
        datos.mean_symmetry, datos.mean_fractal_dimension
    ]])
    
    prediccion = modelo.predict(entrada)
    resultado = "Positivo para cáncer (Maligno)" if prediccion[0] else "Negativo para cáncer (Benigno)"
    
    return {"diagnostico": resultado, "threshold": modelo.threshold}