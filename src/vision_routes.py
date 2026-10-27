"""Endpoints de visao computacional, independentes do modelo de sensores."""
from pathlib import Path
import tempfile
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

try:
    from .visao_computacional import analisar_imagem, comparar_com_gabarito
except ImportError:
    from visao_computacional import analisar_imagem, comparar_com_gabarito

BASE = Path(__file__).resolve().parents[1]
PASTA_RESULTADOS = BASE / "imagens" / "resultados_web"
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
router = APIRouter()

@router.post("/vision/inspect")
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


@router.post("/vision/compare")
async def comparar_imagens(gabarito: UploadFile = File(...),
                           imagem: UploadFile = File(...)):
    """Compara a imagem enviada com um gabarito visual considerado bom."""
    tipos_permitidos = {"image/jpeg", "image/png", "image/webp"}
    uploads = [("gabarito", gabarito), ("imagem", imagem)]
    imagens = {}

    for nome, arquivo in uploads:
        if arquivo.content_type not in tipos_permitidos:
            raise HTTPException(
                status_code=415,
                detail="Envie as duas imagens como JPG, PNG ou WEBP."
            )
        conteudo = await arquivo.read(10 * 1024 * 1024 + 1)
        if len(conteudo) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Cada imagem deve ter no maximo 10 MB."
            )
        imagem_cv = cv2.imdecode(
            np.frombuffer(conteudo, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if imagem_cv is None:
            raise HTTPException(
                status_code=400,
                detail=f"Nao foi possivel ler a imagem {nome}."
            )
        imagens[nome] = imagem_cv

    identificador = uuid.uuid4().hex
    pasta_saida = PASTA_RESULTADOS / identificador
    pasta_saida.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as pasta_temporaria:
        caminho_gabarito = Path(pasta_temporaria) / "gabarito.jpg"
        caminho_analisado = Path(pasta_temporaria) / "imagem_analisada.jpg"
        cv2.imwrite(str(caminho_gabarito), imagens["gabarito"])
        cv2.imwrite(str(caminho_analisado), imagens["imagem"])
        resultado = comparar_com_gabarito(
            caminho_gabarito,
            caminho_analisado,
            pasta_saida,
        )

    resultado["imagens"] = {
        "diferencas_destacadas": f"/resultados/{identificador}/diferencas_destacadas.jpg",
        "mascara_diferenca": f"/resultados/{identificador}/mascara_diferenca.jpg",
        "contornos_gabarito": f"/resultados/{identificador}/gabarito/bordas_contornos.jpg",
        "contornos_analisada": f"/resultados/{identificador}/analisada/bordas_contornos.jpg",
        "hotspots_analisada": f"/resultados/{identificador}/analisada/hotspots_destacados.jpg",
    }
    return resultado