# Clasificación de cáncer de mama: neurona McCulloch-Pitts y FastAPI

**Autor:** Christian Zumárraga  
**Fecha del trabajo inicial:** 14-10-2026  
**Repositorio:** Agregar enlace real de GitHub

## Objetivo

Implementar y consumir una API de predicción académica con una neurona de McCulloch-Pitts entrenada con el conjunto Wisconsin Diagnostic Breast Cancer (WDBC) de scikit-learn.

## Archivos

- `entrenar.py`: partición estratificada, selección estadística de 8 características, binarización, entrenamiento y evaluación.
- `modelo_mp.pkl`: parámetros guardados (nombres, medianas, umbral y etiquetas).
- `api.py`: API FastAPI (`GET /caracteristicas`, `POST /diagnostico`) y servicio del HTML.
- `index.html`: formulario que consulta las 8 características y envía directamente entradas binarias 0 y 1.
- `requirements.txt`: dependencias.

## Instalación y ejecución (terminal integrada de Visual Studio Code)

Abre esta carpeta con Visual Studio Code y abre Terminal > Nueva terminal. Selecciona tu intérprete Python (Conda o venv) y ejecuta:

```bash
python -m pip install -r requirements.txt
python entrenar.py
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Abre **http://127.0.0.1:8000/**. FastAPI también ofrece pruebas de API en **http://127.0.0.1:8000/docs**.

## ¿Cómo funciona?

1. `entrenar.py` separa 75% para entrenamiento y 25% para prueba con `random_state=42`.
2. Utiliza `SelectKBest(f_classif, k=8)` **solo sobre entrenamiento** para elegir ocho variables asociadas estadísticamente con la clase.
3. Calcula la mediana de cada variable seleccionada usando solo entrenamiento. Convierte el valor a `1` si es menor o igual al límite y a `0` si es superior.
4. La neurona M-P suma esas ocho entradas binarias y predice `1` (benigno) si la suma es mayor o igual que su umbral; en caso contrario predice `0` (maligno).
5. El HTML muestra esos límites para orientar la elección de **1 (medida baja)** o **0 (medida alta)**. El usuario envía directamente los 8 valores binarios a FastAPI. Los límites se muestran como referencia educativa; no sustituyen mediciones reales.
6. El frontend recibe JSON y muestra la etiqueta de la clase.

## Ejemplo de petición a `/diagnostico`

Primero consulta `GET /caracteristicas`: los ocho nombres que retorna dependen del entrenamiento. Luego envía exactamente esos nombres con valores **0 o 1**:

```json
{
  "valores": {
    "nombre de característica 1": 1,
    "nombre de característica 2": 0
  }
}
```

**Nota:** el JSON anterior muestra la estructura, no es una petición válida hasta sustituir las claves por las ocho características reales y agregar las ocho entradas binarias.

## Evaluación y limitaciones

El programa imprime exactitud, sensibilidad de la clase maligna y matriz de confusión de prueba. La selección mediante ANOVA F indica asociación estadística, no importancia clínica demostrada. Una neurona M-P de umbral único es un modelo didáctico limitado y puede presentar falsos negativos y falsos positivos. El dataset WDBC contiene medidas extraídas de aspirados con aguja fina; **no admite directamente fotografías, radiografías ni tomografías**. No utilizar el sistema para diagnosticar o atender pacientes.

**Fuente de datos:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
