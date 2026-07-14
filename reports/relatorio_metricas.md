# Relatório de Métricas — Previsão de Risco de AVC

> Gerado automaticamente por `src/train.py`. Não editar à mão.

## 1. Modelo final

- **Algoritmo escolhido:** Regressao Logistica
- **Limiar de decisão:** 0.785 (ajustado na validação, no lugar do padrão 0.5)
- **Balanceamento:** SMOTE aplicado somente no treino, dentro do pipeline
- **Arquivo exportado:** `models/modelo_avc.pkl`

## 2. Comparação dos candidatos (conjunto de validação)

| Modelo | F1 | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| Regressao Logistica | 0.312 | 0.213 | 0.580 | 0.848 |
| Random Forest | 0.234 | 0.159 | 0.440 | 0.801 |
| Gradient Boosting | 0.251 | 0.160 | 0.580 | 0.793 |

## 3. Desempenho final (conjunto de teste)

O teste não foi usado nem para treinar nem para escolher o modelo/limiar.

| Métrica | Valor |
|---|---|
| F1-Score | 0.321 |
| Precision | 0.219 |
| Recall | 0.600 |
| ROC AUC | 0.836 |
| Acurácia | 0.876 |

### Matriz de confusão

|  | Previsto: sem AVC | Previsto: com AVC |
|---|---|---|
| **Real: sem AVC** | 865 | 107 |
| **Real: com AVC** | 20 | 30 |

### Relatório de classificação

```
              precision    recall  f1-score   support

     Sem AVC       0.98      0.89      0.93       972
     Com AVC       0.22      0.60      0.32        50

    accuracy                           0.88      1022
   macro avg       0.60      0.74      0.63      1022
weighted avg       0.94      0.88      0.90      1022

```

## 4. Leitura dos resultados

- **A acurácia engana.** Como apenas ~5% dos pacientes tiveram AVC, prever "ninguém terá
  AVC" já garantiria ~95% de acurácia. Por isso a decisão foi tomada pelo F1 e pelo Recall.
- **ROC AUC de 0.836** indica que o modelo ordena os pacientes por
  risco melhor do que o acaso (0.5), que é o uso realista: priorizar quem investigar primeiro.
- **Recall de 0.600** significa que o modelo encontra
  60% dos pacientes que de fato tiveram AVC. Num contexto
  clínico, deixar de sinalizar um caso real (falso negativo) custa mais caro do que um
  alarme falso, então o Recall foi priorizado frente à Precision.
- **Precision de 0.219** mostra o preço desse ganho: entre os
  pacientes sinalizados, 22% realmente tiveram AVC. O
  restante são alarmes falsos — aceitáveis para uma ferramenta de triagem, que serve para
  indicar quem merece exame adicional, e não para dar diagnóstico.

## 5. Limitações

- O teste tem apenas 50 casos positivos, então as
  métricas da classe minoritária têm margem de erro grande.
- O SMOTE cria pacientes sintéticos por interpolação; ele reduz o viés em favor da classe
  majoritária, mas não acrescenta informação clínica nova.
- Modelo com finalidade acadêmica. **Não deve ser usado para decisão médica real.**
