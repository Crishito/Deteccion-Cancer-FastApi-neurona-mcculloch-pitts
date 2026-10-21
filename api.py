"""API mínima FastAPI para consumir el modelo M-P guardado."""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, StrictInt

DIRECTORIO = Path(__file__).resolve().parent
RUTA_MODELO = DIRECTORIO / "modelo_mp.pkl"
if not RUTA_MODELO.is_file():
    raise RuntimeError("Falta modelo_mp.pkl. Ejecuta primero: python entrenar.py")
modelo = joblib.load(RUTA_MODELO)  # Cargar solo archivos propios y confiables.

app = FastAPI(title="Clasificación de muestras mamarias - Neurona M-P")


class Entrada(BaseModel):
    valores: dict[str, StrictInt]


@app.get("/")
def inicio():
    return FileResponse(DIRECTORIO / "index.html")


@app.get("/caracteristicas")
def caracteristicas():
    return {"caracteristicas": modelo["caracteristicas"], "cortes": modelo["cortes"]}


@app.post("/diagnostico")
def diagnostico(entrada: Entrada):
    nombres = modelo["caracteristicas"]
    recibidas = entrada.valores
    if set(recibidas) != set(nombres):
        raise HTTPException(
            status_code=422,
            detail="Debes enviar exactamente las características solicitadas por /caracteristicas.",
        )
    if any(recibidas[nombre] not in (0, 1) for nombre in nombres):
        raise HTTPException(status_code=422, detail="Cada entrada debe ser 0 o 1.")

    # Las entradas ya son binarias; no se vuelven a transformar.
    entradas_binarias = [recibidas[nombre] for nombre in nombres]
    prediccion = int(sum(entradas_binarias) >= modelo["threshold"])
    return {
        "clasificacion": modelo["clases"][prediccion],
        "clase": prediccion,
        "threshold": modelo["threshold"],
    }
