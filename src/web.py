"""Site AeroMotion e servico de visao, sem dependencia do modelo de sensores."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .vision_routes import PASTA_RESULTADOS, router as vision_router
except ImportError:
    from vision_routes import PASTA_RESULTADOS, router as vision_router

PASTA_SITE = Path(__file__).resolve().parent / "static"
app = FastAPI(title="AeroMotion - site e visao computacional", version="1.0")
app.mount("/static", StaticFiles(directory=PASTA_SITE), name="static")
app.mount("/resultados", StaticFiles(directory=PASTA_RESULTADOS), name="resultados")
app.include_router(vision_router)

@app.get("/", include_in_schema=False)
def pagina_inicial():
    return FileResponse(PASTA_SITE / "index.html")
