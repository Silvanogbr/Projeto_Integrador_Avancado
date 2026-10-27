"preparar os dados dos sensores."
from pathlib import Path
import pandas as pd

SENSORES = [
    "temperatura_mancal_c", "vibracao_rms_mms", "pressao_oleo_bar",
    "rotacao_rpm", "emissao_acustica_db", "corrente_eletrica_amp",
]
FEATURES = SENSORES + ["severidade_dinamica"] + [
    "vibracao_media_1h", "vibracao_desvio_1h", "temperatura_media_1h",
    "temperatura_desvio_1h", "vibracao_media_6h", "vibracao_desvio_6h",
    "temperatura_media_6h", "temperatura_desvio_6h",
]

def preparar_dados(caminho_csv):
    """Le, limpa e cria indicadores temporais por equipamento."""
    df = pd.read_csv(caminho_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["id_equipamento", "timestamp"]).reset_index(drop=True)

    # Remove apenas valores claramente impossiveis/instrumentais do enunciado.
    df.loc[df["temperatura_mancal_c"] > 200, "temperatura_mancal_c"] = float("nan")
    df.loc[df["vibracao_rms_mms"] < 0, "vibracao_rms_mms"] = float("nan")

    # Interpolacao linear dentro de cada bancada; limites completam pontas restantes.
    for coluna in SENSORES:
        df[coluna] = df.groupby("id_equipamento")[coluna].transform(
            lambda serie: serie.interpolate(method="linear", limit_direction="both")
        )
        df[coluna] = df[coluna].fillna(df[coluna].median())

    df["severidade_dinamica"] = df["vibracao_rms_mms"] / df["rotacao_rpm"].replace(0, float("nan"))
    df["severidade_dinamica"] = df["severidade_dinamica"].fillna(0)

    # Uma hora tem 6 leituras e seis horas tem 36 leituras, considerando 10 min.
    for equipamento, indices in df.groupby("id_equipamento").groups.items():
        grupo = df.loc[indices].set_index("timestamp")
        for horas, janela in [(1, "1h"), (6, "6h")]:
            sufixo = f"{horas}h"
            for nome, coluna in [("vibracao", "vibracao_rms_mms"), ("temperatura", "temperatura_mancal_c")]:
                medias = grupo[coluna].rolling(janela, min_periods=1).mean()
                desvios = grupo[coluna].rolling(janela, min_periods=1).std().fillna(0)
                df.loc[indices, f"{nome}_media_{sufixo}"] = medias.to_numpy()
                df.loc[indices, f"{nome}_desvio_{sufixo}"] = desvios.to_numpy()
    return df.sort_values("timestamp").reset_index(drop=True)

def resumo_inspecao(df):
    """Retorna dados basicos para a inspecao estrutural."""
    original = pd.read_csv("dados_sensores_aeromotion.csv")
    faltantes = original.groupby("id_equipamento")[SENSORES].apply(lambda x: x.isna().mean() * 100)
    diferencas = df.groupby("id_equipamento")["timestamp"].diff().dropna()
    return {
        "linhas_colunas": df.shape,
        "tipos": original.dtypes.astype(str).to_dict(),
        "ausentes_pct_por_equipamento": faltantes.round(2),
        "intervalo_mediano_min": diferencas.dt.total_seconds().div(60).median(),
    }
