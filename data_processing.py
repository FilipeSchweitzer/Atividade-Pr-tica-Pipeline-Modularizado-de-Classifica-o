"""
data_processing.py
==================
Módulo de processamento de dados (Módulo A) do pipeline de classificação de churn.

Responsabilidades:
- Ler o arquivo bruto data/raw_transactions.csv (uma linha por transação)
- Agregar as transações por cliente, transformando cada categoria de produto
  em uma coluna com o total gasto (pivot)
- Calcular a coluna 'frequencia' (número de transações do cliente)
- Derivar o rótulo 'churn' (1 se o cliente tem mais de 8 transações)
- Salvar a tabela analítica em data/tabela_churn.csv
- Retornar o DataFrame processado para uso pelo módulo de treinamento

Uso:
    python data_processing.py              # processa e salva a tabela
    python data_processing.py --sem-salvar # processa sem salvar em disco
"""

import os
import argparse
import pandas as pd

# Configurações
CAMINHO_ENTRADA = os.path.join("data", "raw_transactions.csv")
CAMINHO_SAIDA = os.path.join("data", "tabela_churn.csv")
COLUNA_ID = "id_cliente"
COLUNA_CATEGORIA = "categoria"
COLUNA_VALOR = "valor"
COLUNA_FREQUENCIA = "frequencia"
COLUNA_TARGET = "churn"
LIMITE_CHURN = 8   # acima de 8 transações o cliente é marcado como churn


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def carregar_transacoes(caminho=CAMINHO_ENTRADA):
    """Carrega o CSV bruto de transações."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de transações não encontrado em '{caminho}'.")
    return pd.read_csv(caminho)


# ---------------------------------------------------------------------------
# Transformação
# ---------------------------------------------------------------------------
def agregar_por_cliente(df_trans):
    """
    Pivota as transações: uma linha por cliente e uma coluna por categoria,
    contendo a soma dos valores gastos (categorias sem gasto viram 0).
    """
    return df_trans.pivot_table(
        index=COLUNA_ID,
        columns=COLUNA_CATEGORIA,
        values=COLUNA_VALOR,
        aggfunc="sum",
        fill_value=0,
    )


def adicionar_frequencia(df_pivot, df_trans):
    """Adiciona a coluna 'frequencia' (quantidade de transações por cliente)."""
    df_pivot = df_pivot.copy()
    df_pivot[COLUNA_FREQUENCIA] = df_trans.groupby(COLUNA_ID).size()
    return df_pivot


def adicionar_target(df_pivot, limite=LIMITE_CHURN):
    """Cria o rótulo 'churn' a partir da frequência de transações."""
    df_pivot = df_pivot.copy()
    df_pivot[COLUNA_TARGET] = (df_pivot[COLUNA_FREQUENCIA] > limite).astype(int)
    return df_pivot


def construir_tabela_churn(df_trans, limite=LIMITE_CHURN):
    """Aplica todas as transformações e devolve a tabela analítica final."""
    df_pivot = agregar_por_cliente(df_trans)
    df_pivot = adicionar_frequencia(df_pivot, df_trans)
    df_pivot = adicionar_target(df_pivot, limite=limite)
    return df_pivot.reset_index()


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------
def salvar_tabela(df, caminho=CAMINHO_SAIDA):
    """Salva a tabela processada em CSV."""
    diretorio = os.path.dirname(caminho)
    if diretorio:
        os.makedirs(diretorio, exist_ok=True)
    df.to_csv(caminho, index=False)
    print(f"Tabela processada salva em: {caminho}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main(salvar=True):
    """Pipeline completo de processamento: carrega, transforma e (opcional) salva."""
    df_trans = carregar_transacoes()
    print(f"Transações carregadas: {len(df_trans)}")

    df_churn = construir_tabela_churn(df_trans)
    print(f"Clientes agregados: {len(df_churn)}")
    print(f"Colunas geradas: {list(df_churn.columns)}")
    print(f"Distribuição do churn:\n{df_churn[COLUNA_TARGET].value_counts().to_string()}")

    if salvar:
        salvar_tabela(df_churn)

    return df_churn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processamento dos dados de transações")
    parser.add_argument("--sem-salvar", action="store_true",
                        help="Não salva a tabela processada em disco")
    args = parser.parse_args()
    main(salvar=not args.sem_salvar)
