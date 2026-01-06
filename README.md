# 📊 Análise de Risco de Crédito

> Uma análise exploratória abrangente de dados de risco de crédito com insights visuais e estatísticos

## 🎯 Sobre o Projeto

Este repositório contém uma análise detalhada de risco de crédito, explorando padrões de inadimplência e características dos clientes. O projeto utiliza técnicas de análise exploratória de dados (EDA) para identificar os principais fatores que influenciam o risco de crédito.

### Objetivos Principais
- Identificar os padrões de inadimplência nos dados
- Analisar a relação entre perfil de risco e taxa de inadimplência
- Explorar correlações entre variáveis financeiras
- Gerar insights sobre alavancagem e juros nos empréstimos
- Criar visualizações para comunicar descobertas

## 📁 Estrutura do Projeto

```
.
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências do projeto
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv   # Dataset original de risco de crédito
│   └── processed/                    # Dados processados (futuro)
├── notebooks/
│   └── 01_Analise_Exploratoria_Risco.ipynb  # Análise exploratória completa
├── outputs/
│   ├── 01_grafico_inadimplencia.png                    # Taxa de inadimplência geral
│   ├── 02_grafico_Dist_Perfil_Risco.png              # Distribuição do perfil de risco
│   ├── 03_tabela_inadimplencia_por_perfil.PNG        # Tabela de inadimplência por perfil
│   ├── 04_grafico_inadimplencia_por_perfil.png       # Gráfico de inadimplência por perfil
│   ├── 05_grafico_inadimplencia_por_alavancagem.png  # Relação inadimplência vs alavancagem
│   ├── 06_tabela_alavancagem_por_emprestimo.PNG      # Alavancagem por tipo de empréstimo
│   ├── 07_grafico_inadimplencia_acumulada_vs_alavancagem.png  # Análise acumulada
│   ├── 08_grafico_alavancagem_por_inadimplencia.png  # Alavancagem vs inadimplência
│   ├── 09_grafico_matriz_corr.png                    # Matriz de correlação
│   ├── 10_boxplot_juros_vs_risco.png                 # Distribuição de juros por risco
│   ├── EDA_Presentation.pdf                           # Apresentação da análise
│   └── Notebook_Copy.pdf                              # Cópia em PDF do notebook
└── src/
    ├── __init__.py
    └── eda_utils.py                 # Utilitários para EDA e análise estatística
```

## 📊 Dados

### Dataset: `credit_risk_dataset.csv`
O conjunto de dados contém informações sobre clientes e suas características de crédito:

**Variáveis principais analisadas:**
- `SeriousDlqin2yrs` / `Target`: Indicador de inadimplência (variável alvo)
- `person_age`: Idade do cliente
- `person_emp_length`: Tempo de emprego
- `person_income`: Renda do cliente
- `person_home_ownership`: Tipo de propriedade residencial
- `person_loan_amount`: Valor do empréstimo
- `loan_int_rate`: Taxa de juros do empréstimo
- `loan_percent_income`: Taxa de juros como percentual da renda
- `loan_status`: Status do empréstimo
- `loan_intent`: Propósito do empréstimo

### Tratamento de Dados
O notebook realiza limpeza e transformação dos dados, incluindo:
- Tratamento de valores ausentes (valores mediana para tempo de emprego)
- Análise de cardinalidade das variáveis
- Identificação de variáveis numéricas e categóricas
- Preparação para análises estatísticas

## 📈 Análises Realizadas

### 1. **Análise de Inadimplência Geral**
   - Taxa geral de inadimplência no portfólio
   - Visualização de distribuição (sim/não)

### 2. **Perfil de Risco**
   - Distribuição dos clientes por perfil de risco
   - Relação entre perfil de risco e taxa de inadimplência
   - Segmentação por classes de risco

### 3. **Alavancagem (Loan-to-Income)**
   - Análise de alavancagem por tipo de empréstimo
   - Relação entre alavancagem e inadimplência
   - Gráficos acumulados para visualização de tendências

### 4. **Análise de Juros**
   - Distribuição de taxas de juros por nível de risco
   - Box plots comparativos
   - Correlação entre juros e risco

### 5. **Análise de Correlação**
   - Matriz de correlação das variáveis numéricas
   - Identificação de relações entre variáveis

## 🔧 Instalação e Setup

### Pré-requisitos
- Python 3.8+
- pip ou conda

### Instalação
1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd analise-de-risco-crediticio
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Abra o Jupyter Notebook:
```bash
jupyter notebook notebooks/01_Analise_Exploratoria_Risco.ipynb
```

## 📚 Dependências

As principais bibliotecas utilizadas são:

- **pandas** (2.3.3): Manipulação de dados
- **numpy** (2.4.0): Computação numérica
- **matplotlib** (3.10.8): Visualizações de gráficos
- **seaborn** (0.13.2): Visualizações estatísticas
- **scikit-learn** (1.8.0): Machine Learning (análises estatísticas)
- **scipy** (1.16.3): Computação científica
- **jupyter** (cliente 8.7.0): Ambiente interativo

Veja [requirements.txt](requirements.txt) para a lista completa.

## 📊 Utilizando os Utilitários

O módulo `src/eda_utils.py` fornece funções auxiliares para análise exploratória:

```python
from src.eda_utils import identificar_variaveis, missing_report

# Identificar tipos de variáveis
numericas, categoricas = identificar_variaveis(df)

# Gerar relatório de valores ausentes
relatorio = missing_report(df, target_names=['SeriousDlqin2yrs'])
```

### Funções Disponíveis
- `identificar_variaveis(df)`: Classifica colunas em numéricas e categóricas
- `missing_report(df, target_names=None)`: Análise detalhada de valores ausentes com insights para a variável alvo
- Função de análise automática de distribuições (contínua desenvolvimento)

## 📝 Como Usar Este Projeto

1. **Executar a análise**: Abra o notebook `01_Analise_Exploratoria_Risco.ipynb` e execute as células em sequência

2. **Visualizar resultados**: Os gráficos e tabelas são salvos na pasta `outputs/`

3. **Consultar apresentação**: Veja `outputs/EDA_Presentation.pdf` para um resumo visual da análise

4. **Estender a análise**: Use as funções em `src/eda_utils.py` para suas próprias análises

## 🎨 Visualizações Geradas

Todos os outputs visuais são armazenados em `outputs/`:

| Arquivo | Descrição |
|---------|-----------|
| `01_grafico_inadimplencia.png` | Taxa geral de inadimplência |
| `02_grafico_Dist_Perfil_Risco.png` | Distribuição de perfil de risco |
| `03_tabela_inadimplencia_por_perfil.PNG` | Tabela com estatísticas por perfil |
| `04_grafico_inadimplencia_por_perfil.png` | Gráfico de inadimplência por perfil |
| `05_grafico_inadimplencia_por_alavancagem.png` | Análise de alavancagem |
| `08_grafico_alavancagem_por_inadimplencia.png` | Alavancagem vs inadimplência |
| `09_grafico_matriz_corr.png` | Matriz de correlação completa |
| `10_boxplot_juros_vs_risco.png` | Distribuição de juros por risco |

## 🔍 Próximas Etapas

- [ ] Modelagem preditiva com machine learning
- [ ] Feature engineering avançado
- [ ] Validação cruzada e seleção de modelos
- [ ] Explicabilidade de modelos (SHAP, LIME)
- [ ] Pipeline de produção

## 📄 Licença

Este projeto é fornecido como está. Sinta-se livre para usar, modificar e distribuir conforme necessário.

## 👥 Autor

Lucas Domingues - Janeiro 2026

## 📞 Contato e Contribuições

Para dúvidas, sugestões ou contribuições:
- 🔗 LinkedIn: [linkedin.com/in/lucasgdpc](https://linkedin.com/in/lucasgdpc)

---

**Última atualização**: Janeiro 2026
