# AeroMotion Dynamics

Projeto acadêmico de monitoramento de sensores e inspeção visual. O sistema tem dois serviços independentes:

- **Site e visão computacional** (`aeromotion-web`): porta `8000`. Mostra a página, compara imagens e executa o OpenCV.
- **API de sensores** (`aeromotion-api`): porta `8001`. Faz previsões com o modelo de classificação.

Se a API de sensores parar, o site e a visão continuam funcionando. O indicador no topo informa o estado da API e desabilita somente a previsão de sensores.

## Recomendado: ligar tudo com Docker Desktop

### 1. Abrir a pasta do projeto

Abra o Docker Desktop e espere ele ficar pronto. Depois abra o PowerShell e entre na pasta onde está este README. Exemplo:

```powershell
cd "C:\Users\gabriel_silva206\Documents\trabalho integrador avançado"
```

Se o projeto estiver em outra pasta, ajuste o caminho.

### 2. Conferir os arquivos e a configuração

O modelo de classificação precisa estar neste caminho:

```text
modelos/modelo_classificacao.joblib
```

Para conferir a configuração do Docker Compose:

```powershell
docker compose config
```

### 3. Construir e iniciar os dois serviços

Use este comando na primeira vez e sempre que mudar código ou dependências:

```powershell
docker compose up --build -d
```

A primeira construção baixa Python e instala as bibliotecas; pode levar alguns minutos. A opção `-d` deixa os serviços em segundo plano.

### 4. Conferir se estão ligados

```powershell
docker compose ps
```

A saída deve mostrar `aeromotion-web` e `aeromotion-api` com estado `Up`.

Teste os endereços pelo PowerShell:

```powershell
Invoke-WebRequest http://127.0.0.1:8000
Invoke-RestMethod http://127.0.0.1:8001/health
```

### 5. Abrir no navegador

- Site: http://127.0.0.1:8000
- Documentação interativa da API de sensores: http://127.0.0.1:8001/docs
- Estado da API de sensores: http://127.0.0.1:8001/health

No site, preencha os sensores para testar a previsão. Para a visão computacional, escolha primeiro uma imagem gabarito de uma peça aprovada e depois uma imagem da peça a analisar.

## Comandos Docker para controlar os serviços

Execute os comandos na pasta do projeto.

### Ver estado e logs

```powershell
# Ver se cada serviço está ligado
docker compose ps

# Acompanhar os logs dos dois serviços
docker compose logs -f

# Acompanhar apenas os logs do site/visão
docker compose logs -f aeromotion-web

# Acompanhar apenas os logs da API de sensores
docker compose logs -f aeromotion-api
```

`Ctrl+C` ao acompanhar `docker compose logs -f` fecha a visualização dos logs; não desliga os serviços.

### Desligar e ligar apenas a API de sensores

```powershell
# Desliga a previsão por sensores; site e visão continuam ligados
docker compose stop aeromotion-api

# Liga novamente a API de sensores
docker compose start aeromotion-api

# Reinicia somente a API de sensores
docker compose restart aeromotion-api
```

O indicador do site deve mudar em até cinco segundos. A página de visão continua em http://127.0.0.1:8000.

### Desligar e ligar apenas o site/visão

```powershell
# Desliga o site e a visão computacional
docker compose stop aeromotion-web

# Liga novamente o site e a visão
docker compose start aeromotion-web
```

### Reiniciar, atualizar ou encerrar tudo

```powershell
# Reinicia os dois serviços
docker compose restart

# Inicia os serviços já construídos
docker compose up -d

# Reconstrói a imagem e inicia os serviços após mudanças no código
docker compose up --build -d

# Para os containers e remove a rede do projeto
docker compose down
```

Para ver o site novamente depois de `docker compose down`, execute `docker compose up -d`. Se tiver alterado código ou dependências, use `docker compose up --build -d`.

## Alternativa: iniciar sem Docker

É necessário manter **dois terminais PowerShell abertos**. Rode os comandos a partir da pasta do projeto.

### 1. Instalar dependências

```powershell
python --version
python -m pip install -r requirements.txt
```

### 2. Conferir ou gerar o modelo

Para a previsão por sensores, precisa existir `modelos/modelo_classificacao.joblib`. Se o arquivo não existir ou quiser treinar um modelo novo, execute:

```powershell
python src/treinar_modelos.py
```

### 3. Terminal 1: iniciar a API de sensores

```powershell
python -m uvicorn src.api:app --reload --port 8001
```

Deixe esse terminal aberto. A API responde em http://127.0.0.1:8001/health.

### 4. Terminal 2: iniciar o site e a visão computacional

Abra outro PowerShell na pasta do projeto e execute:

```powershell
python -m uvicorn src.web:app --reload --port 8000
```

Deixe também esse terminal aberto e abra http://127.0.0.1:8000. Para desligar uma parte, pressione `Ctrl+C` no terminal correspondente. Parar o terminal da API de sensores não deve fechar o terminal do site/visão.

## Como comparar as imagens

1. Escolha uma imagem gabarito de uma peça boa/aprovada.
2. Escolha a imagem da peça que deseja analisar.
3. Use fotos com ângulo, distância e enquadramento parecidos.
4. O site destaca as diferenças e indica **possível problema** quando mais de 8% dos pixels diferem. Abaixo desse valor, mostra **sem problema aparente**.

O limite de 8% é uma regra simples de comparação, não um classificador treinado com imagens de defeitos. Confirme o resultado com uma pessoa. Hotspots detectados pela cor não são medição real de temperatura.

## Endpoints

| Serviço | Rota | Função |
|---|---|---|
| API de sensores (`8001`) | `GET /health` | Estado e versão do modelo |
| API de sensores (`8001`) | `POST /predict` | Prever a classe de uma leitura |
| API de sensores (`8001`) | `POST /predict-batch` | Prever as classes de várias leituras |
| Site/visão (`8000`) | `POST /vision/compare` | Comparar a peça com o gabarito |
| Site/visão (`8000`) | `POST /vision/inspect` | Analisar uma imagem isolada |

## Solução de problemas

### O site ou a API não abre

Confira o estado dos containers e os logs:

```powershell
docker compose ps
docker compose logs --tail 50
```

Tente reiniciar:

```powershell
docker compose restart
```

### O indicador diz que a API de sensores está desligada

Confira se a API está ligada e responda ao health check:

```powershell
docker compose ps aeromotion-api
Invoke-RestMethod http://127.0.0.1:8001/health
```

Se estiver parada:

```powershell
docker compose start aeromotion-api
```

### A porta 8000 ou 8001 já está sendo usada

Confira qual processo está ouvindo nessas portas:

```powershell
Get-NetTCPConnection -LocalPort 8000,8001 -State Listen
```

Se forem containers antigos deste projeto, pare-os com:

```powershell
docker compose down
```

Depois inicie novamente com:

```powershell
docker compose up -d
```

### Reconstruir tudo do zero após mudanças

```powershell
docker compose down
docker compose up --build -d
```
