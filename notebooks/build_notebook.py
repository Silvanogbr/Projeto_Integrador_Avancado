"""Create the commented project notebook from reusable source code."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = [
nbf.v4.new_markdown_cell("""# AeroMotion Dynamics | IA e análise de dados industriais

**Projeto Integrador Avançado** · Etapas 1 e 2 · Python 3.10+

Este notebook reproduz a inspeção, o saneamento, a engenharia temporal, a EDA e a avaliação multiclasse do CSV recebido. O dataset possui 12.000 leituras fornecidas, sem metadados de origem; a validação mede desempenho neste arquivo, não substitui validação de campo. Execute a partir da raiz do projeto. Os artefatos de modelo são gerados em `artifacts/`.

**Reprodutibilidade:** semente 42. **Alvo:** `tipo_falha` (0 normal, 1 desgaste de rolamento, 2 desbalanceamento de rotor, 3 cavitação/superaquecimento)."""),
nbf.v4.new_code_cell("""from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Jupyter may launch with the notebook folder as the working directory.
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src').is_dir())
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline import SENSORS, PHYSICAL, TARGETS, load_and_engineer, train

CSV = Path('data/dados_sensores_aeromotion.csv')
OUT = Path('artifacts')
assert CSV.exists(), f'CSV não encontrado: {CSV.resolve()}'
"""),
nbf.v4.new_markdown_cell("""## 1. Inspeção estrutural

Confirmamos tipos, chaves, duplicidades, distribuição temporal e ausência por equipamento. As amostras brutas são registradas a cada 10 minutos no eixo temporal global. Como as bancadas alternam registros no arquivo, o intervalo por equipamento pode ser maior que 10 minutos; por isso não se deve interpretar cada intervalo individual de equipamento como perda de pacote."""),
nbf.v4.new_code_cell("""raw = pd.read_csv(CSV, parse_dates=['timestamp'])
display(raw.dtypes.to_frame('dtype'))
print('shape:', raw.shape, '| linhas duplicadas:', raw.duplicated().sum())
assert raw['timestamp'].notna().all()
global_steps = raw['timestamp'].drop_duplicates().sort_values().diff().dropna().dt.total_seconds()/60
display(global_steps.describe().to_frame('intervalo_global_min'))
missing_by_equipment = raw.groupby('id_equipamento')[SENSORS].apply(lambda g: g.isna().mean()*100)
display(missing_by_equipment.round(2).style.format('{:.2f}%'))
display(raw['tipo_falha'].value_counts().sort_index().rename(index=dict(enumerate(TARGETS))).to_frame('amostras'))
"""),
nbf.v4.new_markdown_cell("""## 2. Saneamento e continuidade temporal

O dicionário informa faixas normais, não limites absolutos de falha. Usamos limites de plausibilidade deliberadamente amplos para remover interferências impossíveis (ex.: temperaturas próximas de 1000 °C, vibração negativa), preservando assinaturas legítimas de falha. Lacunas curtas internas são interpoladas no tempo por equipamento, sem extrapolar bordas. Para avaliação, a interpolação é feita separadamente dentro do bloco de treino e do bloco de teste para impedir vazamento entre períodos. Lacunas longas/restantes são imputadas pela mediana do treino no pipeline."""),
nbf.v4.new_code_cell("""engineered, audit = load_and_engineer(CSV)
print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
clean = engineered.copy()
for _, idx in clean.groupby('id_equipamento', sort=False).groups.items():
    part = clean.loc[idx].sort_values('timestamp').set_index('timestamp')
    for col in SENSORS:
        part[col] = part[col].interpolate(method='time', limit=3, limit_area='inside')
    clean.loc[clean.index.intersection(idx), SENSORS] = part[SENSORS].to_numpy()
"""),
nbf.v4.new_markdown_cell("""## 3. Features temporais

As janelas usam durações de tempo reais (1 h e 6 h), agrupadas por equipamento e calculadas antes da leitura atual (`shift(1)`). A cadência global do arquivo é 10 min, mas a alternância dos equipamentos torna irregular a distância entre leituras de uma mesma bancada. O desvio padrão fica ausente quando ainda não há pelo menos duas leituras históricas na janela. Severidade dinâmica é vibração RMS dividida por RPM."""),
nbf.v4.new_code_cell("""feature_cols = [c for c in engineered if 'rolling_' in c or c == 'severidade_dinamica']
display(engineered[['timestamp','id_equipamento','temperatura_mancal_c','vibracao_rms_mms',*feature_cols]].head(12))
"""),
nbf.v4.new_markdown_cell("""## 4. Análise exploratória multivariada

Correlação de Pearson é exploratória e não demonstra causalidade. Pairplot usa uma amostra estratificada para manter a execução leve. As comparações por regime dependem de sensores e falhas representados no CSV."""),
nbf.v4.new_code_cell("""numeric = ['temperatura_mancal_c','vibracao_rms_mms','pressao_oleo_bar','rotacao_rpm','emissao_acustica_db','corrente_eletrica_amp','tipo_falha']
plt.figure(figsize=(10,7)); sns.heatmap(clean[numeric].corr(), cmap='vlag', center=0, annot=True, fmt='.2f'); plt.title('Correlação entre sensores e classe'); plt.tight_layout()
plt.show()
"""),
nbf.v4.new_code_cell("""sns.pairplot(clean.sample(min(1200,len(clean)), random_state=42), vars=numeric[:6], hue='modo_operacao', corner=True, plot_kws={'alpha':.35,'s':12})
plt.show()
"""),
nbf.v4.new_code_cell("""regime_summary = clean.groupby('modo_operacao')[numeric[:-1]].agg(['mean','median','std']).round(2)
display(regime_summary)
fig, axes = plt.subplots(1,2,figsize=(12,4))
sns.boxplot(data=clean, x='modo_operacao', y='temperatura_mancal_c', ax=axes[0]); axes[0].set_title('Temperatura por regime')
sns.boxplot(data=clean, x='modo_operacao', y='vibracao_rms_mms', ax=axes[1]); axes[1].set_title('Vibração por regime')
plt.tight_layout(); plt.show()
"""),
nbf.v4.new_markdown_cell("""## 5. Treino temporal, desbalanceamento e avaliação

O corte respeita a ordem temporal (70% treino, 30% teste). Pesos de classe substituem SMOTE, evitando criar vizinhos sintéticos que misturem períodos ou regimes. `RobustScaler` e imputadores são ajustados dentro do pipeline exclusivamente sobre o treino. Dois classificadores de árvores são comparados; o Random Forest é também ajustado por busca aleatória com validação temporal progressiva no treino. O bloco de teste permanece intocado até a avaliação final."""),
nbf.v4.new_code_cell("""metrics = train(CSV, OUT)
pd.DataFrame({name:{k:v for k,v in values.items() if isinstance(v,(int,float))} for name,values in metrics['metrics'].items()}).T
"""),
nbf.v4.new_code_cell("""for name, score in metrics['metrics'].items():
    print('\\n', name, '| melhores parâmetros:', score.get('best_params', 'padrão'))
    print(pd.DataFrame(score['classification_report']).T.round(3))
    cm = np.array(score['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGETS, yticklabels=TARGETS)
    plt.xlabel('Previsto'); plt.ylabel('Real'); plt.title(f'Matriz de confusão: {name}'); plt.tight_layout(); plt.show()
"""),
nbf.v4.new_markdown_cell("""## 6. Importância e anomalias não supervisionadas

O Random Forest fornece importância por redução de impureza, útil para triagem, mas enviesada em favor de variáveis com muitos pontos de corte. A leitura física deve ser confrontada com conhecimento de engenharia. Isolation Forest é ajustado apenas com exemplos normais do treino; seu limiar `contamination='auto'` requer calibração operacional antes de alarmar operadores."""),
nbf.v4.new_code_cell("""chosen = metrics['model']
import joblib
pipe = joblib.load(OUT/'model.joblib')
prep = pipe.named_steps['prep']; forest = pipe.named_steps['model']
names = prep.get_feature_names_out()
importance = pd.Series(forest.feature_importances_, index=names).sort_values(ascending=False).head(15)
display(importance.to_frame('importância'))
importance.sort_values().plot.barh(figsize=(8,5), title='15 variáveis mais importantes'); plt.tight_layout(); plt.show()
"""),
nbf.v4.new_markdown_cell("""## 7. Conclusão e limites

Os resultados são elevados no arquivo fornecido, mas podem refletir assinaturas de falha muito separáveis e distribuição sintética. Exigir validação em novas campanhas, equipamentos, operadores e sensores; usar intervalos de confiança e custo de falso negativo; validar disponibilidade e sincronização temporal; monitorar drift; e manter aprovação humana para manutenção. A Etapa 3 entrega uma API com schemas, código OpenCV e Docker, mas a inspeção visual/ térmica precisa de imagens calibradas e rótulos reais para validação de desempenho."""),
]
nb.cells = cells
nb.metadata = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10+"}}
Path('notebooks').mkdir(exist_ok=True)
nbf.write(nb, 'notebooks/aeromotion_analise.ipynb')
