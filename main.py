"""
main.py
=======
Script orquestrador do pipeline modularizado de classificação de churn.

Responsabilidades:
- Importar as funções dos módulos A (data_processing) e B (model_training)
- Coordenar a execução do pipeline na ordem correta:
    1) Processamento dos dados brutos  -> data/tabela_churn.csv
    2) Treinamento e avaliação do modelo -> models/modelo_churn.joblib
- Realizar a previsão final para um novo cliente de exemplo e
  exibir o resultado no terminal

Uso:
    python main.py                # executa o pipeline completo
    python main.py --sem-salvar   # executa sem gravar CSV/modelo em disco
"""

import sys
import argparse
import pandas as pd

# --- Módulo A: processamento de dados ---
from data_processing import (
    carregar_transacoes,
    construir_tabela_churn,
    salvar_tabela,
)

# --- Módulo B: treinamento do modelo ---
from model_training import (
    preparar_dados,
    dividir_dados,
    treinar_modelo,
    avaliar_modelo,
    salvar_modelo,
    COLUNA_ID,
)

# Cliente de exemplo para a previsão final (gastos por categoria + frequência)
NOVO_CLIENTE = {
    "Alimentos": 480.00,
    "Casa": 1150.00,
    "Eletronicos": 420.00,
    "Livros": 260.00,
    "Roupas": 1230.00,
    "frequencia": 15,
}

ROTULOS = {0: "Não Churn (cliente retido)", 1: "Churn (cliente em risco)"}


def titulo(texto):
    """Imprime um cabeçalho de etapa para deixar o log legível."""
    print("\n" + "=" * 60)
    print(texto)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Etapa 1 - Processamento (Módulo A)
# ---------------------------------------------------------------------------
def executar_processamento(salvar=True):
    """Lê as transações brutas e devolve a tabela analítica de churn."""
    titulo("ETAPA 1/3 - Processamento dos dados (data_processing.py)")

    df_trans = carregar_transacoes()
    print(f"Transações carregadas: {len(df_trans)}")

    df_churn = construir_tabela_churn(df_trans)
    print(f"Clientes agregados: {len(df_churn)}")
    print(f"Colunas geradas: {list(df_churn.columns)}")

    if salvar:
        salvar_tabela(df_churn)

    return df_churn


# ---------------------------------------------------------------------------
# Etapa 2 - Treinamento (Módulo B)
# ---------------------------------------------------------------------------
def executar_treinamento(df_churn, salvar=True):
    """Treina e avalia o modelo a partir da tabela processada na etapa 1."""
    titulo("ETAPA 2/3 - Treinamento do modelo (model_training.py)")

    # A coluna de identificação não é preditiva e por isso é descartada.
    df = df_churn.drop(columns=[COLUNA_ID])

    X, y = preparar_dados(df)
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    modelo = treinar_modelo(X_train, y_train)
    metricas = avaliar_modelo(modelo, X_test, y_test)

    print(f"\nAcurácia: {metricas['acuracia']:.4f}")
    print(f"Matriz de Confusão:\n{metricas['matriz_confusao']}")
    print(f"\nRelatório de Classificação:\n{metricas['relatorio_classificacao']}")

    if salvar:
        salvar_modelo(modelo)

    return modelo, list(X.columns), metricas


# ---------------------------------------------------------------------------
# Etapa 3 - Previsão final
# ---------------------------------------------------------------------------
def montar_dataframe_cliente(dados_cliente, colunas):
    """
    Converte o dicionário do novo cliente em um DataFrame de uma linha,
    respeitando a ordem exata das colunas usadas no treino.
    """
    faltando = [c for c in colunas if c not in dados_cliente]
    if faltando:
        raise ValueError(
            f"Dados do novo cliente incompletos. Faltam as features: {faltando}"
        )
    extras = [c for c in dados_cliente if c not in colunas]
    if extras:
        print(f"Aviso: features ignoradas (não usadas no treino): {extras}")

    return pd.DataFrame([{c: dados_cliente[c] for c in colunas}], columns=colunas)


def prever_novo_cliente(modelo, dados_cliente, colunas):
    """Aplica o modelo treinado ao novo cliente e exibe o resultado."""
    titulo("ETAPA 3/3 - Previsão para um novo cliente")

    X_novo = montar_dataframe_cliente(dados_cliente, colunas)

    print("Dados do novo cliente:")
    for coluna in colunas:
        print(f"  - {coluna}: {dados_cliente[coluna]}")

    classe = int(modelo.predict(X_novo)[0])
    probabilidades = modelo.predict_proba(X_novo)[0]

    print(f"\nPrevisão: {classe} -> {ROTULOS[classe]}")
    print(f"Probabilidade de Não Churn (0): {probabilidades[0]:.2%}")
    print(f"Probabilidade de Churn (1):     {probabilidades[1]:.2%}")

    return classe, probabilidades


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def main(salvar=True, dados_cliente=None):
    """Executa o pipeline completo: processamento -> treino -> previsão."""
    dados_cliente = dados_cliente if dados_cliente is not None else NOVO_CLIENTE

    df_churn = executar_processamento(salvar=salvar)
    modelo, colunas, metricas = executar_treinamento(df_churn, salvar=salvar)
    classe, probabilidades = prever_novo_cliente(modelo, dados_cliente, colunas)

    titulo("Pipeline concluído com sucesso")
    return {
        "modelo": modelo,
        "metricas": metricas,
        "previsao": classe,
        "probabilidades": probabilidades.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline completo de classificação de churn"
    )
    parser.add_argument("--sem-salvar", action="store_true",
                        help="Não grava a tabela processada nem o modelo em disco")
    args = parser.parse_args()

    # Garante que os acentos apareçam corretamente no terminal do Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    main(salvar=not args.sem_salvar)
