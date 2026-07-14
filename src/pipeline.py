import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def carregar_e_limpar_dados(caminho_dados):
    """
    Carrega o CSV bruto e faz a limpeza inicial.
    Remove a coluna 'id' e trata registros únicos que podem bugar o modelo.
    """
    df = pd.read_csv(caminho_dados)
    
    # Removendo o ID que não serve para predição
    df = df.drop('id', axis=1, errors='ignore')
    
    # Existe apenas 1 paciente com gênero 'Other' na base toda. 
    # Remover ele evita problemas na hora de treinar o OneHotEncoder
    df = df[df['gender'] != 'Other']
    
    return df

def criar_preparador_dados():
    """
    Constrói o motor de transformação (Pipeline) que vai lidar com nulos e escalonamento.
    - Numéricos: Imputa a mediana nos nulos (resolve o IMC) e padroniza as escalas.
    - Categóricos: Transforma textos em colunas de 0 e 1 (One-Hot Encoding).
    """
    colunas_numericas = ['age', 'avg_glucose_level', 'bmi']
    colunas_categoricas = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    transformador_numerico = Pipeline(steps=[
        ('imputador', SimpleImputer(strategy='median')),
        ('escalonador', StandardScaler())
    ])

    transformador_categorico = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preparador = ColumnTransformer(transformers=[
        ('num', transformador_numerico, colunas_numericas),
        ('cat', transformador_categorico, colunas_categoricas)
    ], remainder='passthrough')

    return preparador

def aplicar_balanceamento_smote(X, y):
    """
    Aplica o algoritmo SMOTE para criar dados sintéticos da classe minoritária (casos de AVC),
    balanceando o dataset para o treinamento.
    """
    smote = SMOTE(random_state=42)
    X_balanceado, y_balanceado = smote.fit_resample(X, y)
    
    return X_balanceado, y_balanceado