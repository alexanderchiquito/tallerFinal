from fastapi import FastAPI
from scraping import obtener_datos
from model import entrenar_modelo, predecir_precio
from pydantic import BaseModel
from models import Casa

app = FastAPI()

# Endpoint para obtener datos de web scraping
@app.get("/scraping/")
def scraping():
    df = obtener_datos()
    return {"message": "Datos obtenidos y guardados en CSV", "total": len(df)}

# Endpoint para entrenar el modelo
@app.get("/entrenar/")
def entrenar():
    model, mse = entrenar_modelo()  # Entrenamos el modelo
    return {
        "message": "Modelo entrenado",
        "error_mse": mse
    }


    
@app.post("/predecir/")
def predecir(casa: Casa):
    precio_estimado = predecir_precio(casa)
    return {"precio_estimado": precio_estimado}