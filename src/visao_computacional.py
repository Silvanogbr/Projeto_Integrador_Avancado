"""Inspecao visual: bordas geometricas e possiveis regioes quentes em HSV."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def analisar_imagem(caminho, pasta_saida="imagens/resultado"):
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        raise FileNotFoundError(
            f"Nao foi possivel abrir a imagem: {caminho}"
        )

    destino = Path(pasta_saida)
    destino.mkdir(parents=True, exist_ok=True)

    # 1. Conversao para escala de cinza e reducao de ruido
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)

    # 2. Deteccao de bordas utilizando Canny
    bordas = cv2.Canny(suave, 50, 150)

    # 3. Identificacao de contornos
    contornos, _ = cv2.findContours(
        bordas,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    resultado = imagem.copy()
    cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 2)

    # 4. Calculo de area, perimetro e circularidade
    dados_contornos = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        if perimetro > 0:
            circularidade = (4 * np.pi * area) / (perimetro ** 2)
        else:
            circularidade = 0

        dados_contornos.append({
            "area_pixels": round(float(area), 2),
            "perimetro_pixels": round(float(perimetro), 2),
            "circularidade": round(float(circularidade), 3)
        })

    # 5. Analise de cores em HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Faixa inicial para possiveis regioes vermelhas/alaranjadas.
    # Ajustar conforme a paleta da imagem termica utilizada.
    inferior = np.array([0, 100, 150])
    superior = np.array([35, 255, 255])

    mascara_hotspots = cv2.inRange(hsv, inferior, superior)

    # 6. Destacar possiveis regioes quentes na imagem original
    imagem_hotspots = imagem.copy()
    imagem_hotspots[mascara_hotspots > 0] = (0, 0, 255)

    # 7. Salvar resultados. A escrita via pathlib tambem funciona com pastas
    # que possuem acentos no caminho (algo que cv2.imwrite pode nao aceitar no Windows).
    imagens_saida = {
        "bordas.jpg": bordas,
        "bordas_contornos.jpg": resultado,
        "mascara_hotspots.jpg": mascara_hotspots,
        "hotspots_destacados.jpg": imagem_hotspots,
    }
    for nome, imagem_saida in imagens_saida.items():
        sucesso, codificada = cv2.imencode(".jpg", imagem_saida)
        if sucesso:
            (destino / nome).write_bytes(codificada.tobytes())

    return {
        "quantidade_contornos": len(contornos),
        "pixels_hotspot": int(cv2.countNonZero(mascara_hotspots)),
        "contornos": dados_contornos,
        "aviso": (
            "Hotspots identificados por cor, sem medicao real de temperatura."
        )
    }

def comparar_com_gabarito(caminho_gabarito, caminho_analisado, pasta_saida,
                          limite_diferenca_pct=8.0):
    """Compara uma peca com uma imagem de referencia considerada boa.

    A regra didatica marca como possivel problema quando mais de 8% dos pixels
    apresentam diferenca visivel. As imagens devem ter enquadramento semelhante.
    """
    referencia = cv2.imread(str(caminho_gabarito))
    analisada = cv2.imread(str(caminho_analisado))
    if referencia is None or analisada is None:
        raise ValueError("Nao foi possivel abrir uma das imagens enviadas.")

    altura, largura = referencia.shape[:2]
    analisada = cv2.resize(analisada, (largura, altura))

    referencia_suave = cv2.GaussianBlur(referencia, (5, 5), 0)
    analisada_suave = cv2.GaussianBlur(analisada, (5, 5), 0)
    diferenca_canais = cv2.absdiff(referencia_suave, analisada_suave)
    # Considera mudanca em qualquer canal para nao ignorar defeitos de cor.
    diferenca = np.max(diferenca_canais, axis=2).astype(np.uint8)
    _, mascara = cv2.threshold(diferenca, 30, 255, cv2.THRESH_BINARY)

    percentual = (cv2.countNonZero(mascara) / mascara.size) * 100
    tem_problema = percentual > limite_diferenca_pct

    destaque = analisada.copy()
    destaque[mascara > 0] = (0, 0, 255)
    destino = Path(pasta_saida)
    destino.mkdir(parents=True, exist_ok=True)
    saidas = {
        "diferenca.jpg": diferenca,
        "diferencas_destacadas.jpg": destaque,
        "mascara_diferenca.jpg": mascara,
    }
    for nome, imagem_saida in saidas.items():
        sucesso, codificada = cv2.imencode(".jpg", imagem_saida)
        if sucesso:
            (destino / nome).write_bytes(codificada.tobytes())

    analise_gabarito = analisar_imagem(caminho_gabarito, destino / "gabarito")
    analise_candidata = analisar_imagem(caminho_analisado, destino / "analisada")

    return {
        "tem_problema": bool(tem_problema),
        "resultado": "Possivel problema identificado" if tem_problema else "Sem problema aparente",
        "percentual_diferenca": round(float(percentual), 2),
        "limite_diferenca_pct": float(limite_diferenca_pct),
        "quantidade_contornos_gabarito": analise_gabarito["quantidade_contornos"],
        "quantidade_contornos_analisada": analise_candidata["quantidade_contornos"],
        "pixels_hotspot_gabarito": analise_gabarito["pixels_hotspot"],
        "pixels_hotspot_analisada": analise_candidata["pixels_hotspot"],
        "aviso": (
            "Resultado inicial por diferenca de pixels. Use imagens com mesmo "
            "enquadramento e confirme a avaliacao com uma pessoa. A analise HSV "
            "nao mede temperatura real."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspecao geometrica e visual de possiveis hotspots."
    )

    parser.add_argument(
        "imagem",
        help="Caminho da imagem a ser analisada"
    )

    args = parser.parse_args()

    resultado = analisar_imagem(args.imagem)

    print("\n=== RESULTADO DA INSPECAO ===")
    print(f"Quantidade de contornos: {resultado['quantidade_contornos']}")
    print(f"Pixels de possiveis hotspots: {resultado['pixels_hotspot']}")

    print("\n=== ANALISE GEOMETRICA ===")

    for numero, contorno in enumerate(resultado["contornos"], start=1):
        print(f"\nContorno {numero}:")
        print(f"  Area: {contorno['area_pixels']} pixels")
        print(f"  Perimetro: {contorno['perimetro_pixels']} pixels")
        print(f"  Circularidade: {contorno['circularidade']}")

    print(f"\nAviso: {resultado['aviso']}")
    print("\nImagens salvas na pasta imagens/resultado/")
