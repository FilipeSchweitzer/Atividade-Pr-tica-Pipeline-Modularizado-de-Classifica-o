"""
model_training.py
=================
Módulo de treinamento de modelo de classificação para prever churn de clientes.

Responsabilidades:
- Ler o arquivo data/tabela_churn.csv
- Preparar features (X) e target (y)
- Dividir os dados em treino (70%) e teste (30%), com estratificação
- Treinar um Pipeline scikit-learn (StandardScaler + LogisticRegression)
- Salvar o modelo treinado em disco para uso posterior (joblib)
- Retornar o modelo treinado e as métricas de avaliação
  (Acurácia e Matriz de Confusão)

Uso:
    python model_training.py              # treina, avalia e salva o modelo
    python model_training.py --sem-salvar # treina e avalia sem salvar
"""

import os
import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configurações
CAMINHO_DADOS = os.path.join("data", "tabela_churn.csv")
CAMINHO_MODELO = os.path.join("models", "modelo_churn.joblib")
COLUNA_TARGET = "churn"
COLUNA_ID = "id_cliente"
TEST_SIZE = 0.30   # 30% para teste / 70% para treino
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar_dados(caminho=CAMINHO_DADOS):
    """Carrega o CSV e remove a coluna de identificação (não preditiva)."""
    df = pd.read_csv(caminho)
    df = df.drop(columns=[COLUNA_ID])
    return df


def preparar_dados(df):
    """Separa features (X) e target (y)."""
    X = df.drop(columns=[COLUNA_TARGET])
    y = df[COLUNA_TARGET]
    return X, y


def dividir_dados(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Divide em treino/teste mantendo a proporção das classes (stratify)."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def criar_modelo(random_state=RANDOM_STATE):
    """
    Cria o Pipeline de classificação:
    - StandardScaler: padroniza as features (essencial para regressão logística)
    - LogisticRegression: classificador linear interpretável
    """
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("classificador", LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )),
    ])
    return modelo


def treinar_modelo(X_train, y_train):
    """Cria, treina e retorna o pipeline."""
    modelo = criar_modelo()
    modelo.fit(X_train, y_train)
    return modelo


# ---------------------------------------------------------------------------
# Persistência (salvar/carregar para uso posterior)
# ---------------------------------------------------------------------------
def salvar_modelo(modelo, caminho=CAMINHO_MODELO):
    """Salva o modelo treinado em disco (joblib)."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em: {caminho}")


def carregar_modelo(caminho=CAMINHO_MODELO):
    """Carrega um modelo previamente salvo."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{caminho}'. "
            "Execute o treinamento primeiro (python model_training.py)."
        )
    return joblib.load(caminho)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def avaliar_modelo(modelo, X_test, y_test):
    """Gera as métricas de avaliação no conjunto de teste."""
    y_pred = modelo.predict(X_test)
    metricas = {
        "acuracia": accuracy_score(y_test, y_pred),
        "matriz_confusao": confusion_matrix(y_test, y_pred).tolist(),
        "relatorio_classificacao": classification_report(
            y_test, y_pred, target_names=["Não Churn (0)", "Churn (1)"]
        ),
    }
    return metricas


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main(salvar=True):
    """Pipeline completo: carrega, divide, treina, avalia e (opcional) salva."""
    df = carregar_dados()
    X, y = preparar_dados(df)
    X_train, X_test, y_train, y_test = dividir_dados(X, y)

    print(f"Total de amostras: {len(df)}")
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    modelo = treinar_modelo(X_train, y_train)
    metricas = avaliar_modelo(modelo, X_test, y_test)

    print(f"\nAcurácia: {metricas['acuracia']:.4f}")
    print(f"Matriz de Confusão:\n{metricas['matriz_confusao']}")
    print(f"\nRelatório de Classificação:\n{metricas['relatorio_classificacao']}")

    if salvar:
        salvar_modelo(modelo)

    return modelo, metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento do modelo de churn")
    parser.add_argument("--sem-salvar", action="store_true",
                        help="Não salva o modelo em disco após o treino")
    args = parser.parse_args()
    main(salvar=not args.sem_salvar)