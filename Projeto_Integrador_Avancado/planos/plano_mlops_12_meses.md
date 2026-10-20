# Plano MLOps de monitoramento por 12 meses

## Operação segura

O classificador é apoio à decisão. Alertas não acionam CLP nem substituem intertravamentos, procedimentos de parada ou inspeção autorizada. Fixar versão do modelo, versão do pré-processamento, assinatura dos artefatos e proveniência dos dados em cada release. Um responsável de engenharia aprova implantação e rollback.

## Sinais e cadência

| Cadência | Sinais | Ação |
|---|---|---|
| Contínua | disponibilidade, esquema, relógio, duplicatas, valores físicos, ausência, latência e versão | colocar leitura inválida em quarentena; fallback para regra de segurança e operação existente |
| Diária | alertas, confiança, rejeição, falsos positivos/negativos confirmados e overrides | revisão por manutenção, registrar causa e resultado observado |
| Semanal | distribuição de entrada e de score por bancada/regime, PSI/KS e alarmes por hora | investigar mudança de processo, sensor ou carga; não retreinar automaticamente |
| Mensal | métricas rotuladas por classe, recall, precisão, F1, calibração e antecedência | relatório de desempenho e análise de grupos/bancadas |
| Trimestral | segurança, acesso, vulnerabilidade, restauração, qualidade de rótulos e incidentes | exercício de resposta e revisão de risco |

## Calendário

1. **Meses 0–1:** baseline congelada, perfil de dados, limites físicos aprovados, teste em sombra e responsáveis nomeados.
2. **Meses 2–3:** observabilidade de entrada e inferência, auditoria de alertas e classificação de incidentes.
3. **Meses 4–6:** primeira revisão de drift e avaliação retrospectiva em conjunto temporal rotulado.
4. **Meses 7–9:** candidato avaliado em paralelo, teste de regressão/segurança, canário e rollback ensaiado.
5. **Meses 10–12:** revisão executiva, auditoria de proteção de dados, validação anual e decisão documentada de renovar, atualizar ou retirar.

## Data drift e concept drift

Data drift inclui alterações nas distribuições dos sensores, ausência, frequência de amostragem e mistura de modos. Concept drift ocorre quando sinais semelhantes passam a representar resultados diferentes por desgaste, manutenção, firmware, processo ou definição de rótulo. Um alarme estatístico inicia investigação; não é evidência automática de degradação do modelo. Confirmar calibração e contexto, obter rótulos confiáveis e medir métricas por classe antes de qualquer atualização.

## Gate de release

Requer validação temporal independente e, quando disponível, em equipamento/campanha não vistos; recall mínimo por classe aprovado pela engenharia (rubrica acadêmica sugere >90%, que não é por si só um critério industrial); limite de falsos alarmes/hora; calibração; análise de erros; revisão humana; cibersegurança; assinatura e procedência do dataset/modelo; plano de rollback; documentação e aceite. O conjunto fornecido não prova generalização. A validação visual aguarda imagens calibradas e rótulos de peças boas/defeituosas.
