# Previsão de Risco de AVC — Métodos Quantitativos

Solução de dados de ponta a ponta sobre o [Healthcare Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset):
do CSV bruto até uma aplicação web que serve o modelo treinado.

A base foi escolhida pelas suas imperfeições, que são justamente o objeto do trabalho:
**201 valores nulos** de IMC, variáveis em escalas totalmente distintas (idade, glicose, IMC)
e um desbalanceamento severo — apenas **4,9%** dos pacientes tiveram AVC.

## Integrantes do grupo

> Nome: Agnes Gonçalves, Matheus Vitor, Nathan Cavalcante e Fabio Alexandre

## Stack utilizada

| Camada | Ferramentas |
|---|---|
| Análise e manipulação | pandas, numpy |
| Visualização exploratória | matplotlib, seaborn, Jupyter |
| Pipeline e modelagem | scikit-learn, imbalanced-learn (SMOTE), joblib |
| Dashboard | Plotly |
| Aplicação web | Django + Django Ninja (API tipada) + HTMX |

Versões fixadas em [requirements.txt](requirements.txt). Testado com Python 3.13.

## Como executar

Todos os comandos partem da raiz do repositório.

### 1. Ambiente

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Treinar o modelo (Fase 3)

Gera `models/modelo_avc.pkl`, `reports/relatorio_metricas.md` e `reports/metricas.json`:

```bash
python src/train.py
```

### 3. Gerar o dashboard (Fase 4)

Gera `reports/dashboard.html`. Exige o passo anterior:

```bash
python src/dashboard.py
```

### 4. Subir a aplicação web (Fase 5)

```bash
cd app
python manage.py runserver
```

Depois abra <http://127.0.0.1:8000>.

| Rota | O que é |
|---|---|
| `/` | Formulário de avaliação de risco (HTMX) |
| `/dashboard` | Painel consolidado da Fase 4 |
| `/api/docs` | Documentação interativa da API, gerada a partir dos tipos |
| `/api/prever` | `POST` — recebe um paciente em JSON, devolve a previsão |
| `/api/saude` | `GET` — health check do modelo carregado |

### 5. Explorar a EDA (Fase 1)

```bash
jupyter notebook notebooks/01_eda_e_insights.ipynb
```

### Exemplo de chamada à API

O campo `bmi` pode ser omitido — o pipeline imputa a mediana aprendida no treino:

```bash
curl -X POST http://127.0.0.1:8000/api/prever \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","age":67,"hypertension":0,"heart_disease":1,
       "ever_married":"Yes","work_type":"Private","Residence_type":"Urban",
       "avg_glucose_level":228.69,"bmi":36.6,"smoking_status":"formerly smoked"}'
```

```json
{"probabilidade": 0.7619475434793005, "percentual": 76.2, "alto_risco": false,
 "faixa_risco": "Moderado", "limiar": 0.7848483057168196, "limiar_percentual": 78.5,
 "modelo": "Regressao Logistica", "imc_imputado": false}
```

## Estrutura do diretório

```
avc-prediction-pipeline/
├── app/                          # Fase 5 — aplicação web
│   ├── config/                   # projeto Django (settings, urls, wsgi)
│   ├── predicao/
│   │   ├── apis.py               # rotas JSON (Django Ninja)
│   │   ├── schemas.py            # contratos tipados de entrada/saída
│   │   ├── servico.py            # carrega o .pkl e prevê (usado pela API e pelo front)
│   │   ├── views.py              # front-end HTMX
│   │   └── templates/predicao/   # formulário + fragmento de resultado
│   └── manage.py
├── data/raw/                     # dataset original, sem alteração
├── models/modelo_avc.pkl         # Fase 3 — modelo exportado
├── notebooks/01_eda_e_insights.ipynb   # Fase 1 — EDA e insights
├── reports/
│   ├── dashboard.html            # Fase 4 — painel Plotly
│   ├── relatorio_metricas.md     # Fase 3 — relatório de métricas
│   └── metricas.json             # métricas em formato consumível
├── src/
│   ├── pipeline.py               # Fase 2 — limpeza, imputação, escala, SMOTE
│   ├── train.py                  # Fase 3 — treino, avaliação e exportação
│   └── dashboard.py              # Fase 4 — geração dos gráficos
└── requirements.txt
```

## Fases e entregáveis

| Fase | Entregável | Onde está |
|---|---|---|
| 1 — EDA | Notebook com insights | [notebooks/01_eda_e_insights.ipynb](notebooks/01_eda_e_insights.ipynb) |
| 2 — Pipeline | Script modularizado | [src/pipeline.py](src/pipeline.py) |
| 3 — Modelo | Relatório + modelo exportado | [reports/relatorio_metricas.md](reports/relatorio_metricas.md), `models/modelo_avc.pkl` |
| 4 — Visuais | Dashboard consolidado | [reports/dashboard.html](reports/dashboard.html) (ou `/dashboard`) |
| 5 — Aplicação | Sistema rodando | [app/](app/) |

## Decisões técnicas

**O SMOTE fica dentro do pipeline, não antes dele.** O encadeamento usa o `Pipeline` do
`imbalanced-learn` em vez do `Pipeline` do Scikit-Learn porque só ele aplica o balanceamento
exclusivamente no treino e o ignora no `predict`. Balancear antes de separar os dados criaria
pacientes sintéticos também na validação e no teste, inflando as métricas artificialmente.

**O modelo exportado carrega a preparação junto.** Imputação, escalonamento e One-Hot Encoding
estão dentro do `.pkl`, então a aplicação web envia o paciente no formato bruto do CSV. Isso
elimina a chance clássica de a API preparar o dado de um jeito diferente do treino.

**O limiar de decisão é 0,785, e não 0,5.** O corte padrão é arbitrário e funciona mal em base
desbalanceada. O limiar foi escolhido maximizando o F1 no conjunto de validação.

**A escolha do modelo nunca tocou o teste.** Os três candidatos foram treinados no treino e
comparados na validação; o teste só foi usado na medição final. O modelo exportado é exatamente
aquele cujas métricas o relatório apresenta.

**A acurácia foi deliberadamente ignorada como critério.** Prever "ninguém terá AVC" já daria
~95% de acurácia e seria inútil. A decisão foi tomada por F1, Recall e ROC AUC.

**As cores dos gráficos foram validadas para daltonismo** (separação CVD ΔE ≥ 12, tons claro e
escuro), e todo gráfico traz rótulo direto: a cor nunca é o único canal de informação.

## Resultados

Modelo final: **Regressão Logística** com SMOTE, medida em 1.022 pacientes de teste
nunca vistos no treino.

| Métrica | Valor |
|---|---|
| ROC AUC | 0,836 |
| Recall | 0,600 |
| F1-Score | 0,321 |
| Precision | 0,219 |
| Acurácia | 0,876 |

**Como ler esses números.** O ROC AUC de 0,836 mostra que o modelo ordena pacientes por risco
bem melhor que o acaso — é o uso realista da ferramenta: priorizar quem investigar primeiro.
O Recall de 0,60 significa que 60% dos casos reais de AVC são sinalizados. A Precision de 0,22
é o preço disso: a cada 5 alertas, cerca de 1 se confirma. O trade-off foi escolhido de
propósito, porque num contexto clínico deixar de sinalizar um caso real custa mais caro que um
alarme falso — e porque a base tem só 249 casos positivos para aprender.

Comparação completa e matriz de confusão em [reports/relatorio_metricas.md](reports/relatorio_metricas.md).

---

Projeto acadêmico da disciplina de Métodos Quantitativos.

