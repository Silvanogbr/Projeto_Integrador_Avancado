# Projeto Integrador Avançado - AeroMotion Dynamics

Pacote de análise industrial com notebook, relatório técnico, modelos, API e pipeline OpenCV.

## Entregáveis

- `notebooks/aeromotion_analise.ipynb`: inspeção, saneamento, features temporais, EDA e avaliação dos modelos; notebook executado.
- `output/pdf/relatorio_tecnico_aeromotion.pdf`: relatório técnico da Etapa 1.
- `output/pdf/parecer_tecnico_etico_regulatorio.pdf`: parecer de quatro páginas da Etapa 3.
- `artifacts/model.joblib`, `artifacts/isolation_forest.joblib`, `artifacts/metrics.json`: modelos exportados e métricas.
- `src/pipeline.py`: tratamento e treino temporal.
- `src/vision.py`: inspeção geométrica por Canny e segmentação térmica HSV.
- `src/api.py`: FastAPI com `/predict`, `/predict-batch`, `/health` e `/docs`.
- `Dockerfile`, `docker-compose.yml`, `requirements-api.txt`: implantação em container.
- `docs/plano_mlops_12_meses.md`: plano de monitoramento e governança.

## Reproduzir

Requer Python 3.12. Instale dependências a partir da raiz do projeto:

```bash
python -m pip install -r requirements-dev.txt
python -m src.pipeline data/dados_sensores_aeromotion.csv --out artifacts
python notebooks/build_notebook.py
python scripts/build_reports.py
python -m pytest -q
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Para iniciar o container após gerar os artefatos, use `docker compose up --build`.

## Limites de validação

O modelo usa um corte cronológico único do CSV fornecido: 70% treino e 30% teste. O melhor modelo foi selecionado apenas por validação temporal interna ao treino. O recall por classe de falha no bloco final ficou acima de 90%, próximo do limiar para desgaste de rolamento. Isso não comprova generalização para produção. O CSV não contém imagens, então o código OpenCV foi verificado com imagens sintéticas e ainda requer calibração e dados industriais rotulados. A API mantém janelas móveis de 1 h e 6 h em memória por processo, rejeita leituras fora de ordem e não persiste o histórico após reinício; produção deve compartilhar histórico entre réplicas.
