# Sistema Web de Diagnóstico de Cáncer de Mama (API FastAPI + Neurona M-P)

- **Autor:** Christian Zumárraga
- **Descripción:** Aplicación web completa que despliega una **Neurona de McCulloch y Pitts** entrenada con el dataset de Cáncer de Mama de Wisconsin. Utiliza una API REST construida con **FastAPI** y una interfaz interactiva en **HTML5**.

## Estructura del Proyecto
- `entrenar.py`: Script para procesar los datos, entrenar la neurona con 8 características clave y exportar el modelo (`modelo_mp.pkl`).
- `api.py`: Servidor backend en FastAPI que carga el modelo y expone el endpoint `/diagnostico`.
- `index.html`: Interfaz gráfica para ingresar los valores binarizados y obtener el diagnóstico en tiempo real.
- `requirements.txt`: Dependencias de Python necesarias.

## Instrucciones de Ejecución

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt