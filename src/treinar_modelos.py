from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from data_utils import preparar_dados, FEATURES

BASE = Path(__file__).resolve().parents[1]
MODELOS = BASE / "modelos"
MODELOS.mkdir(exist_ok=True)

def treinar():
    df = preparar_dados(BASE / "dados_sensores_aeromotion.csv")
    # Divisao em ordem temporal: primeiros 70% para treino, ultimos 30% para teste.
    corte = int(len(df) * 0.70)
    X, y = df[FEATURES], df["tipo_falha"]
    X_treino, X_teste = X.iloc[:corte], X.iloc[corte:]
    y_treino, y_teste = y.iloc[:corte], y.iloc[corte:]

    base_rf = Pipeline([("escala", RobustScaler()), ("modelo", RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1))])
    busca = GridSearchCV(base_rf, {
        "modelo__n_estimators": [100, 200],
        "modelo__max_depth": [None, 12],
    }, cv=3, scoring="f1_macro", n_jobs=-1)
    busca.fit(X_treino, y_treino)
    melhor_rf = busca.best_estimator_

    logistico = Pipeline([("escala", RobustScaler()), ("modelo", LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42))])
    logistico.fit(X_treino, y_treino)

    resultados = []
    for nome, modelo in [("Random Forest", melhor_rf), ("Regressao Logistica", logistico)]:
        pred = modelo.predict(X_teste)
        p, r, f, _ = precision_recall_fscore_support(y_teste, pred, average="macro", zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(y_teste, pred, average="weighted", zero_division=0)
        resultados.append({"modelo": nome, "acuracia": accuracy_score(y_teste, pred),
                           "precisao_macro": p, "recall_macro": r, "f1_macro": f,
                           "precisao_weighted": pw, "recall_weighted": rw, "f1_weighted": fw})
        if nome == "Random Forest":
            pd.DataFrame(confusion_matrix(y_teste, pred, labels=[0, 1, 2, 3]),
                         index=["real_"+str(i) for i in range(4)],
                         columns=["previsto_"+str(i) for i in range(4)]).to_csv(BASE / "relatorios" / "matriz_confusao.csv")
            pd.DataFrame(classification_report(y_teste, pred, labels=[0,1,2,3], output_dict=True,
                                               zero_division=0)).T.to_csv(BASE / "relatorios" / "metricas_por_classe.csv")

    pd.DataFrame(resultados).to_csv(BASE / "relatorios" / "comparacao_modelos.csv", index=False)
    joblib.dump(melhor_rf, MODELOS / "modelo_classificacao.joblib")
    # Isolation Forest aprende somente exemplos normais do conjunto de treino.
    normais = X_treino[y_treino == 0]
    isolamento = Pipeline([("escala", RobustScaler()),
                           ("modelo", IsolationForest(contamination=0.05, random_state=42))])
    isolamento.fit(normais)
    joblib.dump(isolamento, MODELOS / "modelo_anomalias.joblib")
    pd.DataFrame({"variavel": FEATURES,
                  "importancia": melhor_rf.named_steps["modelo"].feature_importances_}
                 ).sort_values("importancia", ascending=False).to_csv(
                     BASE / "relatorios" / "importancia_variaveis.csv", index=False)
    print("Melhores parametros:", busca.best_params_)
    print(pd.DataFrame(resultados).to_string(index=False))
    print("Modelos salvos em", MODELOS)

if __name__ == "__main__":
    treinar()
