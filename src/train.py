"""
Fase 3 - Treinamento e Avaliação do Modelo de Previsão de AVC.

Executa o ciclo completo de modelagem:
    1. Carrega e limpa o dado bruto (reaproveitando as funções da Fase 2).
    2. Separa a base em treino, validação e teste de forma estratificada.
    3. Treina modelos candidatos dentro do pipeline (preparação -> SMOTE -> modelo).
    4. Escolhe o melhor modelo e o melhor limiar de decisão usando a VALIDAÇÃO.
    5. Mede o desempenho final no TESTE, que não foi tocado até aqui.
    6. Exporta o modelo final (.pkl) e o relatório de métricas.

Uso:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from pipeline import (
    carregar_e_limpar_dados,
    criar_pipeline_completo,
    separar_features_e_alvo,
)

# Caminhos do projeto, sempre relativos à raiz do repositório
DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS = DIRETORIO_RAIZ / 'data' / 'raw' / 'healthcare-dataset-stroke-data.csv'
DIRETORIO_MODELOS = DIRETORIO_RAIZ / 'models'
DIRETORIO_RELATORIOS = DIRETORIO_RAIZ / 'reports'
CAMINHO_MODELO = DIRETORIO_MODELOS / 'modelo_avc.pkl'
CAMINHO_METRICAS = DIRETORIO_RELATORIOS / 'metricas.json'
CAMINHO_RELATORIO = DIRETORIO_RELATORIOS / 'relatorio_metricas.md'

SEMENTE = 42


def separar_treino_validacao_teste(X, y):
    """
    Divide a base em 60% treino, 20% validação e 20% teste.

    A divisão é estratificada (`stratify=y`) porque só ~5% dos pacientes tiveram AVC:
    sem isso, uma das partições poderia ficar com pouquíssimos casos positivos e
    tornar a avaliação instável.
    """
    # Primeiro corte: separa 20% para teste
    X_restante, X_teste, y_restante, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE
    )

    # Segundo corte: dos 80% restantes, 25% viram validação (= 20% do total)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_restante, y_restante, test_size=0.25, stratify=y_restante, random_state=SEMENTE
    )

    return X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste


def definir_modelos_candidatos():
    """
    Modelos que serão comparados entre si na validação.
    """
    return {
        'Regressao Logistica': LogisticRegression(max_iter=1000, random_state=SEMENTE),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEMENTE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(random_state=SEMENTE),
    }


def encontrar_melhor_limiar(y_verdadeiro, probabilidades):
    """
    Procura o limiar de decisão que maximiza o F1-Score.

    O corte padrão de 0.5 é uma escolha arbitrária e costuma ser ruim em bases
    desbalanceadas. Testar vários cortes permite equilibrar melhor Precision e Recall.
    """
    precisoes, recalls, limiares = precision_recall_curve(y_verdadeiro, probabilidades)

    # A curva devolve um limiar a menos que os pontos de precisão/recall
    precisoes, recalls = precisoes[:-1], recalls[:-1]

    # Fórmula do F1 ponto a ponto, protegida contra divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_por_limiar = np.nan_to_num(2 * (precisoes * recalls) / (precisoes + recalls))

    melhor_indice = int(np.argmax(f1_por_limiar))

    return float(limiares[melhor_indice]), float(f1_por_limiar[melhor_indice])


def avaliar(modelo, X, y, limiar):
    """
    Calcula as métricas adequadas a um problema desbalanceado.

    A acurácia é reportada apenas por completude: um modelo que chutasse "nenhum AVC"
    para todo mundo acertaria ~95% dos casos e ainda assim seria inútil. Por isso o
    que importa aqui é o F1, o Recall e a área sob a curva ROC.
    """
    probabilidades = modelo.predict_proba(X)[:, 1]
    predicoes = (probabilidades >= limiar).astype(int)

    return {
        'acuracia': float(accuracy_score(y, predicoes)),
        'precisao': float(precision_score(y, predicoes, zero_division=0)),
        'recall': float(recall_score(y, predicoes, zero_division=0)),
        'f1': float(f1_score(y, predicoes, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probabilidades)),
        'matriz_confusao': confusion_matrix(y, predicoes).tolist(),
    }


def treinar_e_selecionar(X_treino, y_treino, X_validacao, y_validacao):
    """
    Treina cada candidato no TREINO e escolhe o vencedor pelo F1 na VALIDAÇÃO.

    O teste não é usado em nenhum momento desta função — ele fica reservado para a
    medição final, evitando que a escolha do modelo contamine o resultado reportado.
    """
    resultados = {}
    melhor_nome = None
    melhor_modelo = None
    melhor_limiar = 0.5
    melhor_f1 = -1.0

    for nome, algoritmo in definir_modelos_candidatos().items():
        print(f'  Treinando: {nome}...')

        modelo = criar_pipeline_completo(algoritmo)
        modelo.fit(X_treino, y_treino)

        probabilidades = modelo.predict_proba(X_validacao)[:, 1]
        limiar, f1_validacao = encontrar_melhor_limiar(y_validacao, probabilidades)

        resultados[nome] = avaliar(modelo, X_validacao, y_validacao, limiar)
        resultados[nome]['limiar'] = limiar

        print(
            f'    F1 (validação): {f1_validacao:.3f} | '
            f'ROC AUC: {resultados[nome]["roc_auc"]:.3f} | '
            f'limiar: {limiar:.3f}'
        )

        if f1_validacao > melhor_f1:
            melhor_nome, melhor_modelo, melhor_limiar, melhor_f1 = (
                nome,
                modelo,
                limiar,
                f1_validacao,
            )

    return melhor_nome, melhor_modelo, melhor_limiar, resultados


def montar_relatorio(nome_modelo, limiar, metricas_teste, resultados_validacao, y_teste, predicoes):
    """
    Gera o relatório de métricas em Markdown (entregável da Fase 3).
    """
    verdadeiro_negativo, falso_positivo, falso_negativo, verdadeiro_positivo = (
        np.array(metricas_teste['matriz_confusao']).ravel()
    )

    linhas_comparativo = '\n'.join(
        f'| {nome} | {m["f1"]:.3f} | {m["precisao"]:.3f} | {m["recall"]:.3f} | {m["roc_auc"]:.3f} |'
        for nome, m in resultados_validacao.items()
    )

    return f"""# Relatório de Métricas — Previsão de Risco de AVC

> Gerado automaticamente por `src/train.py`. Não editar à mão.

## 1. Modelo final

- **Algoritmo escolhido:** {nome_modelo}
- **Limiar de decisão:** {limiar:.3f} (ajustado na validação, no lugar do padrão 0.5)
- **Balanceamento:** SMOTE aplicado somente no treino, dentro do pipeline
- **Arquivo exportado:** `models/modelo_avc.pkl`

## 2. Comparação dos candidatos (conjunto de validação)

| Modelo | F1 | Precision | Recall | ROC AUC |
|---|---|---|---|---|
{linhas_comparativo}

## 3. Desempenho final (conjunto de teste)

O teste não foi usado nem para treinar nem para escolher o modelo/limiar.

| Métrica | Valor |
|---|---|
| F1-Score | {metricas_teste['f1']:.3f} |
| Precision | {metricas_teste['precisao']:.3f} |
| Recall | {metricas_teste['recall']:.3f} |
| ROC AUC | {metricas_teste['roc_auc']:.3f} |
| Acurácia | {metricas_teste['acuracia']:.3f} |

### Matriz de confusão

|  | Previsto: sem AVC | Previsto: com AVC |
|---|---|---|
| **Real: sem AVC** | {verdadeiro_negativo} | {falso_positivo} |
| **Real: com AVC** | {falso_negativo} | {verdadeiro_positivo} |

### Relatório de classificação

```
{classification_report(y_teste, predicoes, target_names=['Sem AVC', 'Com AVC'], zero_division=0)}
```

## 4. Leitura dos resultados

- **A acurácia engana.** Como apenas ~5% dos pacientes tiveram AVC, prever "ninguém terá
  AVC" já garantiria ~95% de acurácia. Por isso a decisão foi tomada pelo F1 e pelo Recall.
- **ROC AUC de {metricas_teste['roc_auc']:.3f}** indica que o modelo ordena os pacientes por
  risco melhor do que o acaso (0.5), que é o uso realista: priorizar quem investigar primeiro.
- **Recall de {metricas_teste['recall']:.3f}** significa que o modelo encontra
  {metricas_teste['recall'] * 100:.0f}% dos pacientes que de fato tiveram AVC. Num contexto
  clínico, deixar de sinalizar um caso real (falso negativo) custa mais caro do que um
  alarme falso, então o Recall foi priorizado frente à Precision.
- **Precision de {metricas_teste['precisao']:.3f}** mostra o preço desse ganho: entre os
  pacientes sinalizados, {metricas_teste['precisao'] * 100:.0f}% realmente tiveram AVC. O
  restante são alarmes falsos — aceitáveis para uma ferramenta de triagem, que serve para
  indicar quem merece exame adicional, e não para dar diagnóstico.

## 5. Limitações

- O teste tem apenas {int(verdadeiro_positivo + falso_negativo)} casos positivos, então as
  métricas da classe minoritária têm margem de erro grande.
- O SMOTE cria pacientes sintéticos por interpolação; ele reduz o viés em favor da classe
  majoritária, mas não acrescenta informação clínica nova.
- Modelo com finalidade acadêmica. **Não deve ser usado para decisão médica real.**
"""


def main():
    DIRETORIO_MODELOS.mkdir(exist_ok=True)
    DIRETORIO_RELATORIOS.mkdir(exist_ok=True)

    print('1. Carregando e limpando os dados...')
    df = carregar_e_limpar_dados(CAMINHO_DADOS)
    X, y = separar_features_e_alvo(df)
    print(f'   {len(df)} pacientes | {int(y.sum())} casos de AVC ({y.mean() * 100:.1f}%)')

    print('\n2. Separando em treino, validação e teste...')
    X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = (
        separar_treino_validacao_teste(X, y)
    )
    print(f'   Treino: {len(X_treino)} | Validação: {len(X_validacao)} | Teste: {len(X_teste)}')

    print('\n3. Treinando os modelos candidatos...')
    nome_modelo, modelo, limiar, resultados_validacao = treinar_e_selecionar(
        X_treino, y_treino, X_validacao, y_validacao
    )
    print(f'\n   Modelo vencedor: {nome_modelo} (limiar {limiar:.3f})')

    print('\n4. Avaliando no conjunto de teste...')
    metricas_teste = avaliar(modelo, X_teste, y_teste, limiar)
    print(
        f'   F1: {metricas_teste["f1"]:.3f} | '
        f'Recall: {metricas_teste["recall"]:.3f} | '
        f'Precision: {metricas_teste["precisao"]:.3f} | '
        f'ROC AUC: {metricas_teste["roc_auc"]:.3f}'
    )

    print('\n5. Exportando modelo e relatório...')
    # O objeto salvo carrega o pipeline inteiro (imputação, escalonamento e encoding),
    # então a aplicação da Fase 5 só precisa enviar os dados brutos do paciente.
    joblib.dump(
        {
            'pipeline': modelo,
            'limiar': limiar,
            'nome_modelo': nome_modelo,
            'colunas': list(X.columns),
        },
        CAMINHO_MODELO,
    )

    probabilidades_teste = modelo.predict_proba(X_teste)[:, 1]
    predicoes_teste = (probabilidades_teste >= limiar).astype(int)

    # Guarda também os pontos da curva ROC para os gráficos da Fase 4
    fpr, tpr, _ = roc_curve(y_teste, probabilidades_teste)

    CAMINHO_METRICAS.write_text(
        json.dumps(
            {
                'nome_modelo': nome_modelo,
                'limiar': limiar,
                'validacao': resultados_validacao,
                'teste': metricas_teste,
                'curva_roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )

    CAMINHO_RELATORIO.write_text(
        montar_relatorio(
            nome_modelo, limiar, metricas_teste, resultados_validacao, y_teste, predicoes_teste
        ),
        encoding='utf-8',
    )

    print(f'   Modelo:    {CAMINHO_MODELO.relative_to(DIRETORIO_RAIZ)}')
    print(f'   Métricas:  {CAMINHO_METRICAS.relative_to(DIRETORIO_RAIZ)}')
    print(f'   Relatório: {CAMINHO_RELATORIO.relative_to(DIRETORIO_RAIZ)}')


if __name__ == '__main__':
    main()
