from pathlib import Path
from typing import List
import tempfile
import uuid

import cv2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
try:
    from .data_utils import FEATURES
    from .visao_computacional import analisar_imagem
except ImportError:
    from data_utils import FEATURES
    from visao_computacional import analisar_imagem

BASE = Path(__file__).resolve().parents[1]
PASTA_SITE = Path(__file__).resolve().parent / "static"
PASTA_RESULTADOS = BASE / "imagens" / "resultados_web"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AeroMotion - API de previsao", version="1.0")
app.mount("/static", StaticFiles(directory=PASTA_SITE), name="static")
app.mount("/resultados", StaticFiles(directory=PASTA_RESULTADOS), name="resultados")
modelo = joblib.load(BASE / "modelos" / "modelo_classificacao.joblib")
versao = "1.0"

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
    dados["severidade_dinamica"] = dados["vibracao_rms_mms"] / dados["rotacao_rpm"]
    entrada = pd.DataFrame([dados], columns=FEATURES)
    return {"classe_prevista": int(modelo.predict(entrada)[0])}

@app.get("/", include_in_schema=False)
def pagina_inicial():
    return FileResponse(PASTA_SITE / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "versao_modelo": versao}

@app.post("/predict")
def predict(leitura: Leitura):
    return prever(leitura)

@app.post("/predict-batch")
def predict_batch(lote: Lote):
    return {"resultados": [prever(leitura) for leitura in lote.leituras]}


@app.post("/vision/inspect")
async def inspecionar_imagem(arquivo: UploadFile = File(...)):
    """Recebe uma imagem, executa a analise OpenCV e devolve as imagens."""
    tipos_permitidos = {"image/jpeg", "image/png", "image/webp"}
    if arquivo.content_type not in tipos_permitidos:
        raise HTTPException(status_code=415, detail="Envie uma imagem JPG, PNG ou WEBP.")

    conteudo = await arquivo.read(10 * 1024 * 1024 + 1)
    if len(conteudo) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="A imagem deve ter no maximo 10 MB.")

    imagem = cv2.imdecode(np.frombuffer(conteudo, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise HTTPException(status_code=400, detail="Nao foi possivel ler essa imagem.")

    identificador = uuid.uuid4().hex
    pasta_saida = PASTA_RESULTADOS / identificador
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # O script de visao ja trabalha com caminhos; usamos um arquivo temporario.
    with tempfile.TemporaryDirectory() as pasta_temporaria:
        caminho_temporario = Path(pasta_temporaria) / "imagem.jpg"
        cv2.imwrite(str(caminho_temporario), imagem)
        resultado = analisar_imagem(caminho_temporario, pasta_saida)

    resultado["imagens"] = {
        nome: f"/resultados/{identificador}/{nome}"
        for nome in ["bordas.jpg", "bordas_contornos.jpg", "mascara_hotspots.jpg", "hotspots_destacados.jpg"]
    }
    return resultado
