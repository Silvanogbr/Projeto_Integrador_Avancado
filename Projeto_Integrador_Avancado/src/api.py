"""FastAPI inference service; run after training src.pipeline."""
from __future__ import annotations
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
app = FastAPI(title="AeroMotion Predictive Maintenance API", version=MODEL_VERSION)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
history = defaultdict(lambda: deque(maxlen=200))

def _naive_utc(ts: datetime) -> datetime:
    return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo is not None else ts

class Reading(BaseModel):
    timestamp: datetime
    id_equipamento: str = Field(pattern=r"^TURB-10[1-4]$")
    modo_operacao: str = Field(pattern=r"^(Cruzeiro|Carga_Media|Carga_Alta)$")
    horas_operacao_acumuladas: float = Field(ge=0)
    temperatura_mancal_c: float | None = Field(default=None, ge=20, le=180)
    vibracao_rms_mms: float | None = Field(default=None, ge=0, le=20)
    pressao_oleo_bar: float | None = Field(default=None, ge=0, le=8)
    rotacao_rpm: float | None = Field(default=None, gt=0, le=4000)
    emissao_acustica_db: float | None = Field(default=None, ge=20, le=100)
    corrente_eletrica_amp: float | None = Field(default=None, ge=0, le=50)

def _features(reading: Reading) -> pd.DataFrame:
    d = reading.model_dump(); ts = d.pop("timestamp")
    ts = _naive_utc(ts)
    records = history[reading.id_equipamento]
    if records and ts <= records[-1][0]:
        raise ValueError("Timestamp must be later than the last reading for this equipment")
    for sensor in ["temperatura_mancal_c", "vibracao_rms_mms"]:
        for win in ["1h", "6h"]:
            period = timedelta(hours=int(win[:-1]))
            past = [item for item in records if timedelta(0) < ts-item[0] <= period]
            position = 1 if sensor == "temperatura_mancal_c" else 2
            values = [item[position] for item in past if item[position] is not None]
            d[f"{sensor}_rolling_mean_{win}"] = float(np.mean(values)) if values else np.nan
            d[f"{sensor}_rolling_std_{win}"] = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
    d["severidade_dinamica"] = (d["vibracao_rms_mms"] / d["rotacao_rpm"]
                                  if d["vibracao_rms_mms"] is not None and d["rotacao_rpm"] is not None else np.nan)
    d["hora_dia"] = ts.hour + ts.minute/60
    d["dia_semana"] = ts.weekday()
    records.append((ts, d["temperatura_mancal_c"], d["vibracao_rms_mms"]))
    return pd.DataFrame([d])

@app.get("/health")
async def health():
    return {"status": "ok" if model is not None else "model_not_loaded", "model_version": MODEL_VERSION}

@app.post("/predict")
async def predict(reading: Reading):
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded; train first")
    try: x = _features(reading)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    pred, scores = await run_in_threadpool(_infer_one, x)
    return {"tipo_falha": pred, "classe": ["Normal","Desgaste_Rolamento","Desbalanceamento_Rotor","Cavitacao_Superaquecimento"][pred], "probabilidades": scores, "model_version": MODEL_VERSION}

@app.post("/predict-batch")
async def predict_batch(readings: List[Reading]):
    if not readings: raise HTTPException(status_code=422, detail="Batch cannot be empty")
    if len(readings) > 1000: raise HTTPException(status_code=413, detail="Maximum batch size is 1000")
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded; train first")
    for equipment in {r.id_equipamento for r in readings}:
        times = [_naive_utc(r.timestamp) for r in readings if r.id_equipamento == equipment]
        last = history[equipment][-1][0] if history[equipment] else None
        if any(a >= b for a, b in zip(times, times[1:])) or (last is not None and times[0] <= last):
            raise HTTPException(status_code=422, detail="Batch timestamps must be unique, ascending, and later than prior readings per equipment")
    try: xs = pd.concat([_features(r) for r in readings], ignore_index=True)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    preds = await run_in_threadpool(lambda: model.predict(xs).astype(int).tolist())
    return {"predictions": preds, "model_version": MODEL_VERSION}

def _infer_one(x: pd.DataFrame) -> tuple[int, list[float]]:
    pred = int(model.predict(x)[0])
    scores = model.predict_proba(x)[0].tolist() if hasattr(model, "predict_proba") else []
    return pred, scores
