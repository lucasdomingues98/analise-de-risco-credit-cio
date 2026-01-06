# eda_utils.py
"""
eda_utils.py
Ferramentas de EDA e inferência estatística com mini-relatórios automáticos.
Feito para uso em Jupyter local com matplotlib + seaborn.
"""

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pandas.api.types import is_numeric_dtype

sns.set(style="whitegrid")


# ---------------------------
# Utilitários de identificação
# ---------------------------
def identificar_variaveis(df):
    """
    Retorna duas listas: (colunas_numéricas, colunas_categóricas)
    """
    numericas = df.select_dtypes(include=np.number).columns.tolist()
    categoricas = df.select_dtypes(exclude=np.number).columns.tolist()
    return numericas, categoricas

# Análise automática de missing values e sugestões
def missing_report(df, target_names=None, sample_rows=500, top_n=20):
    # resumo por coluna
    resumo_miss = pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique()
    }).sort_values("pct_missing", ascending=False)
    
    # detectar possível coluna target
    target = None
    possible_targets = (target_names or ['target', 'TARGET', 'Target', 'SeriousDlqin2yrs', 'default'])
    for t in possible_targets:
        if t in df.columns:
            target = t
            break

    # relação missing x target
    if target is not None:
        diffs = {}
        groups = df[target].dropna().unique()
        for col in df.columns:
            try:
                gb = df.groupby(target)[col].apply(lambda s: s.isna().mean()*100)
                if gb.shape[0] >= 2:
                    diffs[col] = (gb.max() - gb.min())
                else:
                    diffs[col] = 0.0
            except Exception:
                diffs[col] = 0.0
        resumo_miss['missing_diff_by_target_pct'] = pd.Series(diffs)
        resumo_miss['missing_target_flag'] = resumo_miss['missing_diff_by_target_pct'].apply(lambda x: x>5)
    else:
        resumo_miss['missing_diff_by_target_pct'] = np.nan
        resumo_miss['missing_target_flag'] = False

    # sugestões simples por coluna
    def suggest(row):
        p = row['pct_missing']
        dtype = row['dtype']
        nunique = row['n_unique']
        if p == 0:
            return "Sem missing"
        if p > 50:
            if np.issubdtype(dtype, np.number):
                return "Alta ausência (>50%): considerar remover coluna ou imputação avançada (model-based). Criar indicador se manter."
            else:
                return "Alta ausência (>50%): considerar remover ou tratar como categoria 'Missing'."
        if p > 10:
            if np.issubdtype(dtype, np.number):
                return "Ausência moderada: imputação (mediana) + indicador; avaliar Iterative/KNN."
            else:
                return "Ausência moderada: imputação (modo) ou nova categoria 'Missing' + indicador."
        # p <= 10
        if np.issubdtype(dtype, np.number):
            return "Baixa ausência: imputação simples (mediana) ou remover linhas; usar pipeline para evitar vazamento."
        else:
            if nunique < 20:
                return "Baixa ausência: imputar com modo ou nova categoria 'Missing'."
            else:
                return "Baixa ausência em categorical com alta cardinalidade: considerar 'Missing' como categoria ou imputação por modelo."

    resumo_miss['sugestao'] = resumo_miss.apply(suggest, axis=1)
    # exibir top N colunas com mais missing
    display(resumo_miss.head(top_n))
    
    # plot barras
    plt.figure(figsize=(10,6))
    sns.barplot(x=resumo_miss['pct_missing'].head(top_n), y=resumo_miss.head(top_n).index)
    plt.xlabel("Percentual de Missing (%)")
    plt.title("Top {} colunas por Missing %".format(top_n))
    plt.show()
    
    # heatmap (amostra para evitar gráfico gigante)
    rows = min(len(df), sample_rows)
    sample = df.sample(rows, random_state=42)
    plt.figure(figsize=(12,6))
    sns.heatmap(sample.isnull(), cbar=False, yticklabels=False)
    plt.title("Mapa de Missing (amostra de {} linhas)".format(rows))
    plt.show()
    
    return resumo_miss


# ---------------------------
# Resumos
# ---------------------------
def resumo_geral(df, head=5):
    """
    Mostra informações gerais do dataset, incluindo:
    - Dimensões e tipos
    - Valores ausentes
    - Cardinalidade e categorias
    - Estatísticas descritivas
    - Outliers via IQR (excluindo variáveis binárias/indicadoras)
    - Primeiras linhas
    """

    import numpy as np
    import pandas as pd
    from IPython.display import display

    print("=== INFORMAÇÕES GERAIS ===")
    print(f"Dimensões: {df.shape}")
    print("\nTipos de dados:")
    display(df.dtypes)

    # -------------------------
    # MISSING
    # -------------------------
    print("\n=== VALORES AUSENTES ===")
    miss_abs = df.isna().sum()
    miss_rel = (df.isna().mean() * 100).round(2)
    missing_table = pd.DataFrame({"N_missing": miss_abs, "%_missing": miss_rel})
    display(missing_table[missing_table["N_missing"] > 0])

    # -------------------------
    # CARDINALIDADE
    # -------------------------
    print("\n=== CARDINALIDADE ===")
    display(df.nunique().sort_values())

    # -------------------------
    # VALORES ÚNICOS POR CATEGÓRICAS
    # -------------------------
    print("\n=== VALORES ÚNICOS (CATEGÓRICAS) ===")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        print(f"\n> {c} ({df[c].nunique()} categorias):")
        display(df[c].value_counts().head(10))

    # -------------------------
    # DESCRITIVAS
    # -------------------------
    print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
    display(df.describe(include="all").T)

    # -------------------------
    # OUTLIERS (IQR) — corrigido para excluir binárias
    # -------------------------
    print("\n=== DETECÇÃO DE OUTLIERS (IQR) ===")

    num_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if df[col].nunique() > 2    # exclui binárias (0/1)
    ]

    outlier_summary = []

    for col in num_cols:
        serie = df[col].dropna()
        Q1, Q3 = serie.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = serie[(serie < lower) | (serie > upper)]
        n_out = len(outliers)
        perc_out = round(n_out / len(serie) * 100, 2)

        outlier_summary.append([col, n_out, perc_out, lower, upper])

    outlier_df = pd.DataFrame(outlier_summary,
        columns=["Variável", "N_outliers", "%_outliers", "Limite_inf", "Limite_sup"]
    )

    display(outlier_df.sort_values("%_outliers", ascending=False))

    # -------------------------
    # HEAD
    # -------------------------
    print(f"\n=== PRIMEIRAS {head} LINHAS ===")
    display(df.head(head))


# ---------------------------
# Visualizações (univariadas)
# ---------------------------

def univariate_analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,    
                              outliers=False, kde=False, color='#8d0801', figsize=(24, 12)):
    '''
    Generate plots for univariate analysis.

    This function generates histograms, horizontal bar plots 
    and boxplots based on the provided data and features. 

    Args:
        data (DataFrame): The DataFrame containing the data to be visualized.
        features (list): A list of feature names to visualize.
        histplot (bool, optional): Generate histograms. Default is True.
        barplot (bool, optional): Generate horizontal bar plots. Default is False.
        mean (bool, optional): Generate mean bar plots of specified feature instead of proportion bar plots. Default is None.
        text_y (float, optional): Y coordinate for text on bar plots. Default is 0.5.
        outliers (bool, optional): Generate boxplots for outliers visualization. Default is False.
        kde (bool, optional): Plot Kernel Density Estimate in histograms. Default is False.
        color (str, optional): The color of the plot. Default is '#8d0801'.
        figsize (tuple, optional): The figsize of the plot. Default is (24, 12).

    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        # Get num_features and num_rows and iterating over the sublot dimensions.
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0) 
        
        fig, axes = plt.subplots(num_rows, 3, figsize=figsize)  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if barplot:
                if mean:
                    data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    bars = ax.barh(y=data_grouped[feature], width=data_grouped[mean], color=color)
                    for index, value in enumerate(data_grouped[mean]):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=15)
                else:
                    data_grouped = data.groupby([feature])[[feature]].count().rename(columns={feature: 'count'}).reset_index()
                    data_grouped['pct'] = round(data_grouped['count'] / data_grouped['count'].sum() * 100, 2)
                    bars = ax.barh(y=data_grouped[feature], width=data_grouped['pct'], color=color)
                    for index, value in enumerate(data_grouped['pct']):
                        # Adjust the text position based on the width of the bars
                        ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=15)
                
                ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=15)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.grid(False)
                ax.get_xaxis().set_visible(False)
                
            elif outliers:
                # Plot univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax, color=color)

            else:
                # Plot histplot.
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='percent')

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
        # Remove unused axes.
        if num_features < len(axes.flat):
            for j in range(num_features, len(axes.flat)):
                fig.delaxes(axes.flat[j])

        plt.tight_layout()
    
    except Exception as e:
        raise CustomException(e, sys)

def plot_distribuicao_numericas(df, coluna, bins=20):
    """
    Histograma + KDE + Boxplot para coluna numérica (aceita também Int64Dtype do pandas).
    """
    if coluna not in df.columns:
        raise KeyError(f"Coluna '{coluna}' não existe no DataFrame.")

    # Usa o verificador do pandas, que reconhece Int64Dtype, Float64Dtype etc.
    if not is_numeric_dtype(df[coluna]):
        raise TypeError(f"Coluna '{coluna}' não é numérica. Use plot_categorica para categóricas.")

    # Converte para float para garantir compatibilidade com seaborn/matplotlib
    serie = df[coluna].astype(float).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(serie, bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f"Histograma - {coluna}")

    sns.boxplot(x=serie, ax=axes[1])
    axes[1].set_title(f"Boxplot - {coluna}")

    plt.tight_layout()
    plt.show()


def tabela_freq_categoricas(df, col):
    """
    Cria uma tabela com a distribuição absoluta e relativa de uma variável
    categórica ou numérica discreta (com poucos valores únicos).

    Compatível com tipos pandas como Int64 e Float64.
    """
    serie = df[col].dropna()

    # Função auxiliar para detectar numéricos de forma segura
    def is_numeric(series):
        return pd.api.types.is_numeric_dtype(series)

    # Detecta variável numérica discreta (<= 20 valores únicos)
    if is_numeric(serie) and serie.nunique() <= 20:
        tipo = "numérica discreta"
        freq_abs = pd.DataFrame(serie.value_counts().sort_index())
        freq_rel = round(pd.DataFrame(serie.value_counts(normalize=True).sort_index()) * 100, 2).astype(str) + "%"
    else:
        tipo = "categórica"
        freq_abs = pd.DataFrame(serie.value_counts().sort_index())
        freq_rel = round(pd.DataFrame(serie.value_counts(normalize=True).sort_index()) * 100, 2).astype(str) + "%"

    # Junta as duas tabelas
    tab = pd.concat([freq_abs, freq_rel], axis=1)
    tab.columns = ['Frequência Absoluta', 'Frequência Relativa']

    # Exibe cabeçalho e tabela
    print("=" * 70)
    print(f"Tabela de Frequências da variável: {col} ({tipo})")
    print("=" * 70)
    display(tab)
    print("\n")

    return tab


# ---------------------------
# Visualizações (bivariadas)
# ---------------------------

def bivariada_categ_categ(df, alvo):
    """
    Cria gráficos para analisar a relação entre uma variável alvo categórica
    e todas as variáveis categóricas/discretas do dataframe.
    Inclui legendas com contraste e rótulos percentuais nas barras empilhadas.

    Parâmetros:
        df (pd.DataFrame): dataframe de entrada
        alvo (str): nome da coluna alvo (categórica)
    """
    
    # Seleciona colunas categóricas ou discretas (poucos valores únicos)
    cols_categoricas = [
        col for col in df.columns 
        if col != alvo and 
        (df[col].dtype == 'object' or df[col].nunique() <= 10)
    ]
    
    if not cols_categoricas:
        print("Nenhuma variável categórica/discreta encontrada.")
        return
    
    n = len(cols_categoricas)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(14, 4 * nrows))

    # Paleta com bom contraste — azul e laranja
    cores = sns.color_palette(["#1f77b4", "#ff7f0e"])

    for i, col in enumerate(cols_categoricas, 1):
        plt.subplot(nrows, ncols, i)
        
        data = df.copy()
        
        # Ordena e converte variáveis discretas
        if data[col].dtype != 'object':
            data = data.sort_values(by=col)
            data[col] = data[col].astype(str)
        else:
            data[col] = data[col].astype(str)
        
        # Plotagem da proporção
        sns.histplot(
            data=data, 
            x=col, 
            hue=alvo, 
            multiple='fill', 
            shrink=0.8,
            palette=cores,
            edgecolor='white'
        )

        # Rótulos percentuais
        ax = plt.gca()
        for c in ax.containers:
            labels = [f'{(v.get_height())*100:.1f}%' if v.get_height() > 0 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=9, fontweight='bold')

        plt.title(f"{alvo} por {col}", fontsize=12, fontweight='bold')
        plt.ylabel("Proporção")
        plt.xlabel(col)
        plt.ylim(0, 1)
        
        # === Correção: legenda manual ===
        categorias_alvo = sorted(df[alvo].unique())
        legend_handles = [Patch(facecolor=cores[i], label=f"{alvo} = {cat}") for i, cat in enumerate(categorias_alvo)]
        plt.legend(handles=legend_handles, loc='upper right', title=alvo)

    plt.tight_layout()
    plt.show()
    

def bivariada_categoricas(df, categoricas, var_alvo):

    melted = pd.melt(
        df,
        id_vars=[var_alvo],
        value_vars=categoricas,
        var_name="variaveis"
    )

    # garantir que rótulos apareçam
    melted["value"] = melted["value"].astype(str)

    g = sns.catplot(
        data=melted,
        x="value",
        y=var_alvo,
        col="variaveis",
        col_wrap=3,
        kind="box",
        height=4.5,        # AUMENTA O ESPAÇO DO FACET
        aspect=1.1,        # MAIS LARGO
        sharex=False,      # EVITA REMOVER TICKS ENTRE FACETS
        sharey=False
    )

    g.set_axis_labels("Categorias", var_alvo)
    g.set_titles("{col_name}")

    # ajuda a evitar sobreposição
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.90)

    g.fig.suptitle(f"Relação entre {var_alvo} e variáveis categóricas")

    plt.show()



def bivariada_numericas(df, numericas, alvo):

    melted = pd.melt(df, id_vars=[alvo], value_vars=numericas)

    def scatter_loess(data, color="#1f77b4", **kws):
        ax = plt.gca()

        # remove NA
        d = data[["value", alvo]].dropna()

        # scatter
        sns.scatterplot(
            data=d, x="value", y=alvo,
            s=22, alpha=0.7, color=color, ax=ax
        )

        # critério seguro para ativar LOESS:
        # - mais de 10 pontos
        # - ao menos 6 valores distintos no eixo X
        # - desvio padrão não nulo
        if (
            len(d) > 10 and
            d["value"].nunique() > 6 and
            np.std(d["value"]) > 0
        ):
            lo = lowess(d[alvo], d["value"], frac=0.4, return_sorted=True)
            ax.plot(lo[:, 0], lo[:, 1], color="red", linewidth=2)

    # FacetGrid grande e bonito
    g = sns.FacetGrid(
        melted,
        col="variable",
        col_wrap=4,
        height=4,       # aumenta tamanho do gráfico
        aspect=1.2,
        sharex=False,
        sharey=False
    )

    g.map_dataframe(scatter_loess)

    g.set_titles("{col_name}")
    g.set_axis_labels("value", alvo)
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(f"Relação entre {alvo} e variáveis numéricas", fontsize=18)

    plt.show()


def bivariada_numericas_categorica(df, numericas, alvo):

    melted = pd.melt(
        df,
        id_vars=[alvo],
        value_vars=numericas,
        var_name="variavel"
    )

    melted[alvo] = melted[alvo].astype(str)

    g = sns.catplot(
        data=melted,
        x=alvo,
        y="value",
        hue=alvo,
        col="variavel",
        col_wrap=3,
        kind="box",
        height=4.5,
        aspect=1.2,
        sharex=False,
        sharey=False,
        legend=True
    )

    g.set_axis_labels("Categorias", "Valor")
    g.set_titles("{col_name}")

    # =======================================================
    #        ⬆️ AUMENTA ESPAÇO SUPERIOR PARA TÍTULO+LEGENDA
    # =======================================================
    g.fig.subplots_adjust(top=0.78)

    # =======================================================
    #              TÍTULO GERAL (NO TOPO ABSOLUTO)
    # =======================================================
    g.fig.suptitle(
        f"Relação entre {alvo} e variáveis numéricas",
        fontsize=17,
        y=0.98           # bem no topo
    )

    # =======================================================
    #              LEGENDA GLOBAL (ABAIXO DO TÍTULO)
    # =======================================================
    leg = g._legend
    if leg is not None:
        leg.set_title(alvo)

        # AQUI ESTÁ A MÁGICA: usar loc=upper center e
        # mover usando bbox_to_anchor (sem mexer em _loc)
        leg.set_bbox_to_anchor((0.5, 0.88))
        leg.set_frame_on(True)

    plt.show()



def analise_bivariada(df, alvo, numericas, categoricas=None, alvo_numerica=True):
    if alvo_numerica:
        if numericas:
            bivariada_numericas(df=df, numericas=numericas, alvo=alvo)
        elif categoricas:
            bivariada_categoricas(df=df, categoricas=categoricas, var_alvo=alvo)
        else:
            print("Variáveis independentes não especificadas. Determine o tipo: categóricas ou numéricas")
    else:
        if categoricas:
            bivariada_categ_categ(df=df, alvo=alvo)
        else:
            bivariada_numericas_categorica(df, numericas, alvo)




'''
def plotar_numericas_completo(df):
    """Plota distribuições de todas as variáveis numéricas."""
    num_cols, _ = identificar_variaveis(df)
    n = len(num_cols)
    if n == 0:
        print("Nenhuma variável numérica encontrada.")
        return
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribuição de {col}")
    plt.tight_layout()
    plt.show()


def plotar_categoricas_completo(df):
    """Plota distribuições de todas as variáveis categóricas."""
    _, cat_cols = identificar_variaveis(df)
    n = len(cat_cols)
    if n == 0:
        print("Nenhuma variável categórica encontrada.")
        return
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette="pastel", ax=axes[i])
        axes[i].set_title(f"Frequência de {col}")
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, hue=None, diag_kind='kde', palette='pastel', height=2.5):
    """
    Plota um pairplot estilizado para as variáveis numéricas de um DataFrame.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com as variáveis numéricas.
    hue : str, opcional
        Coluna categórica para colorir os pontos.
    diag_kind : str, opcional
        Tipo de gráfico na diagonal ('hist' ou 'kde').
    palette : str ou dict, opcional
        Paleta de cores.
    height : float, opcional
        Altura de cada subplot.
    """
    # Seleciona apenas colunas numéricas
    df_num = df.select_dtypes(include='number')
    
    # Estilo bonito do seaborn
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    # Cria o pairplot
    pairplot = sns.pairplot(df_num, hue=hue, diag_kind=diag_kind,
                            palette=palette, height=height, corner=False,
                            plot_kws={'alpha':0.7, 's':40, 'edgecolor':'k'})
    
    # Ajustes finais
    pairplot.fig.suptitle("Pairplot das Variáveis Numéricas", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
'''

# ---------------------------
# Testes de pressupostos
# ---------------------------
def teste_normalidade(df, colunas, alpha=0.05):
    """
    Teste de Shapiro-Wilk para uma ou várias colunas.
    - aceita string (uma coluna) ou lista de colunas.
    - imprime mini-relatório para cada coluna e retorna DataFrame com resultados.
    """
    if isinstance(colunas, str):
        colunas = [colunas]
    results = []
    for col in colunas:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada.")
        vals = df[col].dropna()
        if len(vals) < 3:
            stat, p = np.nan, np.nan
            decision = "Amostra pequena (<3) - teste não aplicável"
        else:
            stat, p = stats.shapiro(vals)
            decision = "Não rejeita H0 (aparenta normalidade)" if p > alpha else "Rejeita H0 (não normal)"
        # print mini-relatório
        print(f"\n=== Teste de Normalidade (Shapiro-Wilk) — '{col}' ===")
        print("H0: amostra segue distribuição normal.")
        print("H1: amostra NÃO segue distribuição normal.")
        print(f"Estatística W = {stat:.4f}" if not np.isnan(stat) else "Estatística W = nan")
        print(f"p-valor = {p:.4f}" if not np.isnan(p) else "p-valor = nan")
        print(f"Decisão (α={alpha}): {decision}")
        results.append({"variavel": col, "estatistica": stat, "p_valor": p, "decisao": decision})
    return pd.DataFrame(results)


def teste_homogeneidade_variancias(*grupos, metodo="levene", alpha=0.05):
    """
    Teste de homogeneidade de variâncias.
    - aceita séries/arrays com os grupos passados diretamente:
        teste_homogeneidade_variancias(grp1, grp2, grp3, metodo='levene')
    - metodo: 'levene' (robusto) ou 'bartlett' (requer normalidade)
    - imprime mini-relatório e retorna dicionário com estatística/p-valor/decisão
    """
    metodo = metodo.lower()
    if metodo == "levene":
        stat, p = stats.levene(*grupos)
        name = "Levene"
    elif metodo == "bartlett":
        stat, p = stats.bartlett(*grupos)
        name = "Bartlett"
    else:
        raise ValueError("metodo deve ser 'levene' ou 'bartlett'")
    decision = "Não rejeita H0 (variâncias homogêneas)" if p > alpha else "Rejeita H0 (variâncias diferentes)"
    # mini-relatório
    print(f"\n=== Teste de Homogeneidade de Variâncias ({name}) ===")
    print("H0: variâncias populacionais são iguais")
    print("H1: pelo menos uma variância é diferente")
    print(f"Estatística = {stat:.4f}")
    print(f"p-valor = {p:.4f}")
    print(f"Decisão (α={alpha}): {decision}")
    return {"teste": name, "estatistica": stat, "p_valor": p, "decisao": decision}

def pressupostos(df, colunas, alfa=0.05):
    """ Testa os dois pressupostos para o conjunto de dados selecionado: normalidade e homogeneidade de variâncias """
    teste_normalidade(df, colunas, alpha=alfa)
    print()
    teste_homogeneidade_variancias(*[df[col] for col in colunas], metodo='levene', alpha=alfa)
    
    


def teste_homogeneidade_variancias_df(df, valor_col, grupo_col, metodo="levene", alpha=0.05):
    """
    Versão amigável: recebe DataFrame e nomes de colunas (valor, grupo).
    Separa automaticamente os grupos e chama teste_homogeneidade_variancias.
    """
    groups = []
    missing_groups = []
    for g in df[grupo_col].unique():
        arr = df.loc[df[grupo_col] == g, valor_col].dropna()
        if arr.size == 0:
            missing_groups.append(g)
        else:
            groups.append(arr)
    if missing_groups:
        print(f"[!] grupos sem observações e ignorados: {missing_groups}")
    if len(groups) < 2:
        raise ValueError("É necessário pelo menos dois grupos com observações.")
    return teste_homogeneidade_variancias(*groups, metodo=metodo, alpha=alpha)

def analise_levene(df, colunas=None, alfa=0.05):
    """
    Aplica o teste levene para comparar as variâncias de diferentes grupos
    """
    alfa_percent = str(round(100*alfa,2)) + "%"
    if colunas is None:
        estat, p_valor = stats.levene(
            *[df[col] for col in df.columns], nan_policy="omit")
    else:
        estat, p_valor = stats.levene(
            df[colunas], nan_policy="omit")
    print(f"=== Teste de Homogeneidade das Variâncias ===")
    print(f"H0: As amostras possuem variâncias iguais.")
    print(f"H1: As amostras não possuem variâncias iguais.")
    print(f"Estatística calculada=({estat:.3f}), p-valor=({p_valor:.4f}).")
    if p_valor < alfa:
        print(f"Hipótese nula (H0) rejeitada. Variâncias são diferentes a {round(100*alfa,2)}% de significância.")
    else:
        print(f"Hipótese nula (H0) aceita. Variâncias não são diferentes a {alfa_percent} de significância.")
    print()

def analise_shapiro(df, col=None, alfa=0.05):
    """
    Aplica o teste de normalidade para as colunas especificadas de um dataframe.
    Assume todas as colunas do dataframe, se não forem especificadas.
    """
    estat, p_valor = stats.shapiro(df[col], nan_policy="omit")
    print(f"==== Teste de Normalidade (Shapiro) da variável {col} ====")
    print(f"H0: {col} segue uma distribuição normal.")
    print(f"H1: {col} não segue uma distribuição normal.")
    alfa_percent = str(round(100*alfa,2)) + "%"
    print(f"Estatística calculada=({estat:.3f}), p-valor=({p_valor:.4f}).")
    if p_valor < alfa:
        print(f"Rejeita-se a hipótese nula (H0) a uma significância de {alfa_percent}. Não segue uma distribuição normal!")
    else:
        print(f"Não se rejeita a hipótese nula (H0) a uma significância de {alfa_percent}. Aparenta seguir uma distribuição normal!")
    print()

def analise_shapiro_levene(df, colunas=None, alfa=0.05):
    """ Testa normalidade e homogeneidade das variâncias dos dados selecionados.
    """
    analise_shapiro(df, colunas, alfa=alfa)
    analise_levene(df, colunas, alfa=alfa)


# ---------------------------
# Testes de hipótese (duas amostras)
# ---------------------------
def teste_t_independente(grupo1, grupo2, equal_var=False, alternative="two-sided", alpha=0.05):
    """
    Teste t para duas amostras independentes.
    - equal_var=False -> Welch (não assume variâncias iguais)
    - alternative: 'two-sided', 'less', 'greater' (scipy >=1.9 supports alternative)
    - imprime mini-relatório e retorna dict
    """
    # SciPy older versions might not support 'alternative' in ttest_ind; handle fallback
    try:
        stat, p = stats.ttest_ind(grupo1, grupo2, equal_var=equal_var, alternative=alternative)
    except TypeError:
        # fallback: get two-sided p then convert
        stat, p_two = stats.ttest_ind(grupo1, grupo2, equal_var=equal_var)
        if alternative == "two-sided":
            p = p_two
        else:
            # determine direction by stat sign
            if alternative == "greater":
                p = p_two / 2 if stat > 0 else 1 - p_two / 2
            else:  # 'less'
                p = p_two / 2 if stat < 0 else 1 - p_two / 2

    h1_text = {
        "two-sided": "As médias são diferentes",
        "greater": "Média(grupo1) > Média(grupo2)",
        "less": "Média(grupo1) < Média(grupo2)"
    }[alternative]

    decision = "Rejeita H0" if p < alpha else "Não rejeita H0"

    print("\n=== Teste t (independente) ===")
    print("H0: médias populacionais são iguais")
    print(f"H1: {h1_text}")
    print(f"Estatística t = {stat:.4f}")
    print(f"p-valor = {p:.4f} (α={alpha})")
    print(f"Decisão: {decision}")
    return {"teste": "t-independente", "estatistica": stat, "p_valor": p, "decisao": decision}


def teste_t_pareado(before, after, alternative="two-sided", alpha=0.05):
    """
    Teste t pareado (before, after series).
    """
    try:
        stat, p = stats.ttest_rel(before, after, alternative=alternative)
    except TypeError:
        stat, p_two = stats.ttest_rel(before, after)
        if alternative == "two-sided":
            p = p_two
        else:
            if alternative == "greater":
                p = p_two / 2 if stat > 0 else 1 - p_two / 2
            else:
                p = p_two / 2 if stat < 0 else 1 - p_two / 2

    h1_text = {
        "two-sided": "As médias são diferentes",
        "greater": "Média(before) > Média(after)",
        "less": "Média(before) < Média(after)"
    }[alternative]

    decision = "Rejeita H0" if p < alpha else "Não rejeita H0"
    print("\n=== Teste t pareado ===")
    print("H0: média das diferenças = 0")
    print(f"H1: {h1_text}")
    print(f"Estatística t = {stat:.4f}")
    print(f"p-valor = {p:.4f} (α={alpha})")
    print(f"Decisão: {decision}")
    return {"teste": "t-pareado", "estatistica": stat, "p_valor": p, "decisao": decision}


def teste_mannwhitney(grupo1, grupo2, alpha=0.05):
    """
    Mann-Whitney U (não paramétrico para 2 amostras independentes).
    """
    stat, p = stats.mannwhitneyu(grupo1, grupo2, alternative="two-sided")
    decision = "Rejeita H0" if p < alpha else "Não rejeita H0"
    print("\n=== Teste Mann-Whitney U ===")
    print("H0: distribuições são iguais")
    print("H1: distribuições são diferentes")
    print(f"U = {stat:.4f}")
    print(f"p-valor = {p:.4f}")
    print(f"Decisão: {decision}")
    return {"teste": "Mann-Whitney", "estatistica": stat, "p_valor": p, "decisao": decision}

def teste_mannwhitney_proprio(grupo1, grupo2, alfa=0.05, alternative='two-sided'):
    """
    Teste para duas amostras independentes (não paramétrico)
    """
    estat, p_valor = stats.mannwhitneyu(grupo1, grupo2, alternative=alternative, nan_policy="omit")
    print(f"=== Teste Mann-Whitney-U ===")
    h0 = "H0: As distribuições são iguais."
    h1 = {
        "two-sided": "As distribuições não são iguais",
        "greater": "A distribuição do grupo 1 > grupo 2",
        "less": "A distribuição do grupo 1 < grupo 2" 
    }[alternative]
    print(h0)
    print(f"H1: {h1}.")
    print(f"Estatística=({estat:.3f}), p-valor=({p_valor:.4f})")
    if p_valor < alfa:
        print(f"H0 rejeitada a {round(100*alfa,2)}% de significância.")
    else:
        print(f"H0 aceita a {round(100*alfa,2)}% de significância.")


def teste_wilcoxon(before, after, modo='two-sided', alfa=0.05):
    """
    Teste não paramétrico para comparar as médias de duas amostras pareadas.
    """
    before = before.dropna()
    after = after.dropna()
    alpha = alfa
    alfa = round(100*(alfa),2)
    estat, p_valor = stats.wilcoxon(before, after, alternative=modo)
    tam = len("=== Teste Wilcoxon (não paramétrico) para amostras pareadas ===")
    print(f"="*tam)
    print(f"=== Teste Wilcoxon (não paramétrico) para amostras pareadas ===")
    print(f"H0: as médias são iguais a {alfa}% de significância.")
    if modo=='two-sided':
        print(f"H1: As médias são diferentes a {alfa}% de significância.")
        decis = "Médias são diferentes"
    elif modo=='greater':
        print(f"H1: Média antes > média depois a {alfa}% de significância.")
    else:
        print(f"H1: Média antes < média depois a {alfa}% de significância.")
    decisao = "H0 aceita. Médias são iguais!" if p_valor>alpha else "H0 rejeitada"
    print(f"Estatística=({estat:.3f}), p-valor=({p_valor:.4f})")
    print(f"Decisão: {decisao}.")
    print(f"="*tam)


# ---------------------------
# Testes (3+ grupos)
# ---------------------------
def anova_oneway_df(df, valor_col, grupo_col, alpha=0.05):
    """
    ANOVA one-way (paramétrica) a partir de DataFrame.
    """
    groups = [df.loc[df[grupo_col] == g, valor_col].dropna() for g in df[grupo_col].unique()]
    stat, p = stats.f_oneway(*groups)
    decision = "Rejeita H0 (alguma média difere)" if p < alpha else "Não rejeita H0"
    print("\n=== ANOVA one-way ===")
    print("H0: todas as médias são iguais")
    print(f"F = {stat:.4f}")
    print(f"p-valor = {p:.4f}")
    print(f"Decisão: {decision}")
    return {"teste": "ANOVA", "estatistica": stat, "p_valor": p, "decisao": decision}


def kruskal_oneway_df(df, valor_col, grupo_col, alpha=0.05):
    """
    Kruskal-Wallis (não paramétrico, 3+ grupos) a partir de DataFrame.
    """
    groups = [df.loc[df[grupo_col] == g, valor_col].dropna() for g in df[grupo_col].unique()]
    stat, p = stats.kruskal(*groups)
    decision = "Rejeita H0 (pelo menos uma distribuição difere)" if p < alpha else "Não rejeita H0"
    print("\n=== Kruskal-Wallis ===")
    print("H0: distribuições populacionais são iguais")
    print(f"H = {stat:.4f}")
    print(f"p-valor = {p:.4f}")
    print(f"Decisão: {decision}")
    return {"teste": "Kruskal-Wallis", "estatistica": stat, "p_valor": p, "decisao": decision}

def teste_friedman(*grupos, nan_policy="omit", alfa=0.05):
    """
    Teste não paramétrico para 3+ grupos não pareados
    """
    estat, p_valor = stats.friedmanchisquare(*grupos, nan_policy="omit")
    print(f"==== Teste Friedman ====")
    print(f"H0: As medianas são iguais a {100*alfa:.1f}% de significância.")
    print(f"H1: As medianas são diferentes a {100*alfa:.1f}% de significância.")
    print(f"Estatística=({estat:.3f}), p-valor=({p_valor:.4f})")
    if p_valor > alfa:
        print(f"H0 aceita. As medianas são iguais!")
    else:
        print(f"H0 rejeitada. A mediana de pelo menos 1 grupo é distinta.")

def teste_kruskal(*grupos, alfa=0.05):
    estat, p_valor = stats.kruskal(*grupos, nan_policy="omit")
    print(f"==== Teste Kruskal ====")
    print(f"H0: as medianas são iguais.")
    print(f"H1: a mediana de pelo menos 1 grupo é distinta.")
    print(f"Estatística=({estat:.3f}), p-valor=({p_valor:.4f})")
    if p_valor > alfa:
        print(f"H0 é aceita a {100*alfa:.1f}% de significância. As medianas são iguais!")
    else:
        print(f"H0 é rejeitada a {100*alfa:.1f}% de significância. As medianas não são iguais!")


# ---------------------------
# Teste Qui-quadrado (categóricas)
# ---------------------------
def teste_chi2(df, col1, col2, alpha=0.05):
    """
    Teste de independência Qui-quadrado entre duas variáveis categóricas.
    - imprime tabela de contingência, estatística, p-valor e decisão.
    """
    cont = pd.crosstab(df[col1], df[col2])
    stat, p, dof, expected = stats.chi2_contingency(cont)
    decision = "Rejeita H0 (associação significativa)" if p < alpha else "Não rejeita H0 (sem associação significativa)"
    print("\n=== Teste Qui-quadrado de Independência ===")
    print(f"Tabela de contingência: ({col1} x {col2})")
    display(cont)
    print(f"χ² = {stat:.4f}, dof = {dof}, p-valor = {p:.4f}")
    print(f"Decisão: {decision}")
    return {"teste": "chi2", "estatistica": stat, "p_valor": p, "dof": dof, "decisao": decision, "esperado": expected}



# ---------------------------
# Pequenos helpers
# ---------------------------
def _format_float(x):
    try:
        return float(f"{x:.4f}")
    except Exception:
        return x

