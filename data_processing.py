# %%
import pandas as pd

# %%
df_trans = pd.read_csv('data/raw_transactions.csv')

# %%
df_trans

# %%
df_pivot = df_trans.copy()
df_pivot = df_pivot.pivot_table(index='id_cliente', columns='categoria', values='valor', aggfunc='sum', fill_value=0)
df_pivot['frequencia'] = df_trans.groupby('id_cliente').size()
df_pivot['churn'] = (df_pivot['frequencia'] > 8).astype(int)
df_pivot = df_pivot.reset_index()
df_pivot.to_csv('data/tabela_churn.csv', index=False)


