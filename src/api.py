"""API independente para classificacao de falhas nos sensores."""
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .data_utils import FEATURES
except ImportError:
    from data_utils import FEATURES

BASE = Path(__file__).resolve().parents[1]
modelo = joblib.load(BASE / "modelos" / "modelo_classificacao.joblib")
versao = "1.0"
app = FastAPI(title="AeroMotion - API de sensores", version=versao)
# O site local usa outra porta; liberamos as chamadas do navegador para esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class Leitura(BaseModel):
    temperatura_mancal_c: float = Field(ge=-50, le=200)
    vibracao_rms_mms: float = Field(ge=0, le=100)
    pressao_oleo_bar: float = Field(ge=0, le=20)
    rotacao_rpm: float = Field(gt=0, le=10000)
    emissao_acustica_db: float = Field(ge=0, le=150)
    corrente_eletrica_amp: float = Field(ge=0, le=200)
    vibracao_media_1h: float
    vibracao_desvio_1h: float
    temperatura_media_1h: float
    temperatura_desvio_1h: float
    vibracao_media_6h: float
    vibracao_desvio_6h: float
    temperatura_media_6h: float
    temperatura_desvio_6h: float

class Lote(BaseModel):
    leituras: List[Leitura]

def prever(leitura):
    dados = leitura.model_dump()
    dados["severidade_dinamica"] = (
        dados["vibracao_rms_mms"] / dados["rotacao_rpm"]
    )
    entrada = pd.DataFrame([dados], columns=FEATURES)
    return {"classe_prevista": int(modelo.predict(entrada)[0])}

@app.get("/health")
def health():
    return {"status": "ok", "versao_modelo": versao}

@app.post("/predict")
def predict(leitura: Leitura):
    return prever(leitura)

@app.post("/predict-batch")
def predict_batch(lote: Lote):
    return {"resultados": [prever(leitura) for leitura in lote.leituras]}
