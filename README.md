# Projeto Integrador AeroMotion

## O que foi feito

Esta é uma solução acadêmica simples para o enunciado do PDF. Ela inclui limpeza e análise dos sensores, classificação de falhas, detecção de anomalias, uma demonstração inicial de visão computacional, API e Docker.

## Estrutura

- `notebooks/atividade_aeromotion.ipynb`: notebook comentado em etapas.
- `src/data_utils.py`: limpeza e criação de variáveis.
- `src/treinar_modelos.py`: divisão temporal, comparação e exportação dos modelos.
- `src/visao_computacional.py`: bordas, contornos e máscara HSV para regiões quentes.
- `src/api.py`: API REST, página web e endpoints `/health`, `/predict`, `/predict-batch` e `/vision/inspect`.
- `src/static/index.html`: site para previsão dos sensores e envio de imagens para o OpenCV.
- `modelos/`: modelos `.joblib` gerados pelo treino.
- `relatorios/`: métricas, comparação e matriz de confusão.
- `relatorios/parecer_tecnico.pdf`: relatório técnico e parecer resumido.

## Como executar

1. Instale Python 3.10 ou superior.
2. Na pasta do projeto, execute `pip install -r requirements.txt`.
3. Gere os modelos: `python src/treinar_modelos.py`.
4. Inicie o servidor: `python -m uvicorn src.api:app --reload`.
5. Abra `http://127.0.0.1:8000` para usar o site. A documentação fica em `http://127.0.0.1:8000/docs`.
6. Para iniciar tudo com Docker, use `docker compose up --build` e acesse `http://127.0.0.1:8000`. Para parar o serviço, execute `docker compose down`.
7. No site, envie a leitura dos sensores ou selecione uma imagem JPG, PNG ou WEBP para análise visual. O limite de imagem é 10 MB.
8. A análise visual também pode ser executada pelo terminal: `python src/visao_computacional.py caminho/para/imagem.jpg`.

## Entradas para a API

A leitura inclui os seis sensores e os valores móveis de vibração e temperatura para janelas de 1h e 6h. A severidade dinâmica é calculada pela API. O retorno é o código da classe (0 a 3). O endpoint em lote recebe `{"leituras": [...]}`. O endpoint `/vision/inspect` recebe uma imagem no campo `arquivo` e devolve contornos, possíveis hotspots e os endereços das imagens geradas.

## Observações

Os limites e a faixa HSV são pontos de partida didáticos. A validação de geometria e temperatura depende de imagens calibradas, imagens térmicas reais e medidas de referência. O conjunto é sintético/acadêmico; não se recomenda usar as previsões para comandar máquinas reais sem validação técnica, monitoramento e aprovação de especialistas.
