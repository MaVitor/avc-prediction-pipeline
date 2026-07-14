"""
Fase 4 - Criação de Visuais (Dashboard consolidado).

Reúne, em um único HTML interativo, as características da base (Fase 1), o efeito
do balanceamento (Fase 2) e o desempenho do modelo (Fase 3). O arquivo gerado é
autocontido e também é servido pela aplicação web da Fase 5, em `/dashboard`.

Uso:
    python src/dashboard.py    (exige que src/train.py já tenha sido executado)
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go

from pipeline import (
    aplicar_balanceamento_smote,
    carregar_e_limpar_dados,
    criar_preparador_dados,
    separar_features_e_alvo,
)

DIRETORIO_RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS = DIRETORIO_RAIZ / 'data' / 'raw' / 'healthcare-dataset-stroke-data.csv'
CAMINHO_MODELO = DIRETORIO_RAIZ / 'models' / 'modelo_avc.pkl'
CAMINHO_METRICAS = DIRETORIO_RAIZ / 'reports' / 'metricas.json'
CAMINHO_DASHBOARD = DIRETORIO_RAIZ / 'reports' / 'dashboard.html'

# Paleta validada para daltonismo (ver README). Azul = referência/ausência de AVC,
# vermelho = risco/presença de AVC, verde-água = dado sintético criado pelo SMOTE.
AZUL = '#2a78d6'
VERMELHO = '#e34948'
AGUA = '#1baf7a'
SUPERFICIE = '#fcfcfb'
GRADE = '#e1e0d9'
TINTA_SECUNDARIA = '#52514e'
TINTA_APAGADA = '#898781'
FONTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Escala sequencial de azul (claro = perto de zero, escuro = magnitude alta)
ESCALA_AZUL = [[0.0, '#cde2fb'], [0.5, '#3987e5'], [1.0, '#0d366b']]


def aplicar_estilo(figura, titulo, altura=360):
    """
    Aplica a identidade visual comum a todos os gráficos: eixos discretos,
    grade quase invisível e tipografia do sistema.
    """
    figura.update_layout(
        title=dict(text=titulo, font=dict(size=15, color='#0b0b0b')),
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family=FONTE, size=12, color=TINTA_SECUNDARIA),
        # A altura é fixa, mas a largura acompanha o painel (ver `config` em montar_html).
        # Sem isso o Plotly assume 700px e corta as últimas barras dentro da grade.
        autosize=True,
        height=altura,
        margin=dict(l=60, r=24, t=56, b=48),
        hoverlabel=dict(font_family=FONTE),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0),
    )
    figura.update_xaxes(gridcolor=GRADE, linecolor='#c3c2b7', tickfont=dict(color=TINTA_APAGADA))
    figura.update_yaxes(gridcolor=GRADE, linecolor='#c3c2b7', tickfont=dict(color=TINTA_APAGADA))

    return figura


def grafico_desbalanceamento(y):
    """
    O problema central da base: quase todo paciente é da classe negativa.
    """
    contagem = y.value_counts().sort_index()
    total = int(contagem.sum())
    rotulos = ['Sem AVC', 'Com AVC']

    figura = go.Figure(
        go.Bar(
            x=rotulos,
            y=contagem.values,
            marker_color=[AZUL, VERMELHO],
            # Rótulo direto em cada barra: a cor nunca é o único canal de informação
            text=[f'{v} ({v / total * 100:.1f}%)' for v in contagem.values],
            textposition='outside',
            textfont=dict(color=TINTA_SECUNDARIA),
            hovertemplate='%{x}: %{y} pacientes<extra></extra>',
            width=0.5,
        )
    )
    figura.update_yaxes(title='Pacientes', range=[0, contagem.max() * 1.18])

    return aplicar_estilo(figura, 'Desbalanceamento da base: apenas ~5% tiveram AVC')


def grafico_efeito_smote(X, y):
    """
    Compara a distribuição das classes antes e depois do SMOTE, mostrando por que o
    balanceamento entrou no pipeline.
    """
    preparador = criar_preparador_dados()
    X_preparado = preparador.fit_transform(X)
    _, y_balanceado = aplicar_balanceamento_smote(X_preparado, y)

    antes = y.value_counts().sort_index()
    depois = pd.Series(y_balanceado).value_counts().sort_index()
    rotulos = ['Sem AVC', 'Com AVC']

    figura = go.Figure()
    for nome, serie, cor in [('Antes do SMOTE', antes, AZUL), ('Depois do SMOTE', depois, AGUA)]:
        figura.add_trace(
            go.Bar(
                name=nome,
                x=rotulos,
                y=serie.values,
                marker_color=cor,
                marker_line=dict(color=SUPERFICIE, width=2),
                text=serie.values,
                textposition='outside',
                textfont=dict(color=TINTA_SECUNDARIA),
                hovertemplate=f'{nome}<br>%{{x}}: %{{y}} pacientes<extra></extra>',
            )
        )
    figura.update_layout(barmode='group', bargap=0.35)
    figura.update_yaxes(title='Registros', range=[0, depois.max() * 1.18])

    return aplicar_estilo(figura, 'Efeito do SMOTE: a classe minoritária é igualada no treino')


def grafico_risco_por_idade(df):
    """
    Traduz o insight principal da EDA em números de negócio: a taxa de AVC por faixa etária.
    """
    faixas = pd.cut(
        df['age'],
        bins=[0, 20, 30, 40, 50, 60, 70, 80, 100],
        labels=['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+'],
    )
    taxa = df.groupby(faixas, observed=True)['stroke'].agg(['mean', 'size'])
    taxa['mean'] *= 100

    figura = go.Figure(
        go.Bar(
            x=taxa.index.astype(str),
            y=taxa['mean'],
            # Série única com magnitude crescente: escala sequencial de um só tom
            marker=dict(color=taxa['mean'], colorscale=ESCALA_AZUL, showscale=False),
            text=[f'{v:.1f}%' for v in taxa['mean']],
            textposition='outside',
            textfont=dict(color=TINTA_SECUNDARIA),
            customdata=taxa['size'],
            hovertemplate='Faixa %{x}<br>Taxa de AVC: %{y:.1f}%<br>Pacientes: %{customdata}<extra></extra>',
        )
    )
    figura.update_xaxes(title='Faixa etária')
    figura.update_yaxes(title='% que teve AVC', range=[0, taxa['mean'].max() * 1.18])

    return aplicar_estilo(figura, 'Taxa de AVC por faixa etária: o risco dispara após os 60')


def grafico_curva_roc(metricas):
    """
    Curva ROC do modelo final no conjunto de teste.
    """
    roc = metricas['curva_roc']
    auc = metricas['teste']['roc_auc']

    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Palpite aleatório',
            line=dict(color=TINTA_APAGADA, width=2, dash='dash'),
            hoverinfo='skip',
        )
    )
    figura.add_trace(
        go.Scatter(
            x=roc['fpr'],
            y=roc['tpr'],
            mode='lines',
            name=f'Modelo (AUC = {auc:.3f})',
            line=dict(color=AZUL, width=2),
            fill='tozeroy',
            fillcolor='rgba(42,120,214,0.12)',
            hovertemplate='Falsos positivos: %{x:.2f}<br>Recall: %{y:.2f}<extra></extra>',
        )
    )
    figura.update_xaxes(title='Taxa de falso positivo', range=[0, 1])
    figura.update_yaxes(title='Taxa de verdadeiro positivo (Recall)', range=[0, 1.02])

    return aplicar_estilo(figura, f'Curva ROC no teste — AUC {auc:.3f}')


def grafico_matriz_confusao(metricas):
    """
    Matriz de confusão do teste: mostra onde o modelo erra, e não só quanto ele erra.
    """
    matriz = metricas['teste']['matriz_confusao']
    rotulos_x = ['Previsto: sem AVC', 'Previsto: com AVC']
    rotulos_y = ['Real: sem AVC', 'Real: com AVC']

    figura = go.Figure(
        go.Heatmap(
            z=matriz,
            x=rotulos_x,
            y=rotulos_y,
            colorscale=ESCALA_AZUL,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate='%{y}<br>%{x}<br>%{z} pacientes<extra></extra>',
        )
    )
    # O valor vai escrito em cada célula: o tom sozinho não precisa ser decodificado
    for i, linha in enumerate(matriz):
        for j, valor in enumerate(linha):
            figura.add_annotation(
                x=rotulos_x[j],
                y=rotulos_y[i],
                text=f'<b>{valor}</b>',
                showarrow=False,
                font=dict(size=17, color='#ffffff' if valor > max(map(max, matriz)) * 0.45 else '#0b0b0b'),
            )
    figura.update_yaxes(autorange='reversed')

    return aplicar_estilo(figura, 'Matriz de confusão (conjunto de teste)')


def traduzir_variavel(nome_bruto):
    """
    Converte os nomes técnicos gerados pelo ColumnTransformer
    (ex: 'cat__smoking_status_smokes') em rótulos legíveis.
    """
    nome = nome_bruto.split('__', 1)[-1]

    traducoes = {
        'age': 'Idade',
        'avg_glucose_level': 'Glicose média',
        'bmi': 'IMC',
        'hypertension': 'Hipertensão',
        'heart_disease': 'Doença cardíaca',
        'gender_Male': 'Gênero: masculino',
        'gender_Female': 'Gênero: feminino',
        'ever_married_Yes': 'Já foi casado(a)',
        'ever_married_No': 'Nunca foi casado(a)',
        'Residence_type_Urban': 'Residência: urbana',
        'Residence_type_Rural': 'Residência: rural',
        'work_type_Private': 'Trabalho: privado',
        'work_type_Self-employed': 'Trabalho: autônomo',
        'work_type_Govt_job': 'Trabalho: público',
        'work_type_children': 'Trabalho: criança',
        'work_type_Never_worked': 'Nunca trabalhou',
        'smoking_status_smokes': 'Fuma',
        'smoking_status_formerly smoked': 'Ex-fumante',
        'smoking_status_never smoked': 'Nunca fumou',
        'smoking_status_Unknown': 'Tabagismo desconhecido',
    }

    return traducoes.get(nome, nome)


def grafico_fatores_de_risco(modelo_salvo):
    """
    Coeficientes da Regressão Logística, que indicam o peso de cada variável.

    Só faz sentido para modelos lineares; se o vencedor for baseado em árvores,
    o gráfico é omitido do dashboard.
    """
    pipeline = modelo_salvo['pipeline']
    classificador = pipeline.named_steps['modelo']

    if not hasattr(classificador, 'coef_'):
        return None

    nomes = pipeline.named_steps['preparador'].get_feature_names_out()
    coeficientes = pd.Series(classificador.coef_[0], index=[traduzir_variavel(n) for n in nomes])

    # As 12 variáveis de maior peso, independente do sinal
    principais = coeficientes.reindex(coeficientes.abs().sort_values().index).tail(12)

    figura = go.Figure(
        go.Bar(
            x=principais.values,
            y=principais.index,
            orientation='h',
            # Escala divergente: vermelho aumenta o risco, azul reduz, zero é o neutro
            marker_color=[VERMELHO if v > 0 else AZUL for v in principais.values],
            hovertemplate='%{y}<br>Coeficiente: %{x:.3f}<extra></extra>',
        )
    )
    figura.add_vline(x=0, line_color='#c3c2b7', line_width=2)
    figura.update_xaxes(title='← reduz o risco   |   aumenta o risco →')
    figura.update_layout(margin=dict(l=170, r=24, t=56, b=48))

    return aplicar_estilo(
        figura, 'Peso de cada variável no modelo (coeficientes da Regressão Logística)', altura=420
    )


def grafico_comparacao_modelos(metricas):
    """
    Justifica a escolha do modelo final mostrando o desempenho dos candidatos na validação.
    """
    validacao = metricas['validacao']
    nomes = list(validacao.keys())

    figura = go.Figure()
    for metrica, rotulo, cor in [
        ('f1', 'F1-Score', AZUL),
        ('recall', 'Recall', AGUA),
        ('roc_auc', 'ROC AUC', VERMELHO),
    ]:
        valores = [validacao[n][metrica] for n in nomes]
        figura.add_trace(
            go.Bar(
                name=rotulo,
                x=nomes,
                y=valores,
                marker_color=cor,
                marker_line=dict(color=SUPERFICIE, width=2),
                text=[f'{v:.2f}' for v in valores],
                textposition='outside',
                textfont=dict(color=TINTA_SECUNDARIA),
                hovertemplate=f'{rotulo}<br>%{{x}}: %{{y:.3f}}<extra></extra>',
            )
        )
    figura.update_layout(barmode='group', bargap=0.3)
    figura.update_yaxes(title='Pontuação', range=[0, 1.05])

    return aplicar_estilo(figura, 'Comparação dos candidatos no conjunto de validação')


def montar_cartoes(metricas):
    """
    Monta os indicadores numéricos do topo do painel.
    """
    teste = metricas['teste']
    cartoes = [
        ('ROC AUC', f'{teste["roc_auc"]:.3f}', 'Capacidade de ordenar pacientes por risco'),
        ('Recall', f'{teste["recall"]:.1%}', 'Dos casos reais de AVC, quantos foram detectados'),
        ('F1-Score', f'{teste["f1"]:.3f}', 'Equilíbrio entre Precision e Recall'),
        ('Precision', f'{teste["precisao"]:.1%}', 'Dos alertas emitidos, quantos se confirmaram'),
    ]

    return '\n'.join(
        f'''<div class="cartao">
      <div class="cartao-rotulo">{rotulo}</div>
      <div class="cartao-valor">{valor}</div>
      <div class="cartao-nota">{nota}</div>
    </div>'''
        for rotulo, valor, nota in cartoes
    )


def montar_html(figuras, metricas):
    """
    Junta os cartões e os gráficos em uma página única e autocontida.
    """
    # O Plotly entra embutido no primeiro gráfico para que o HTML funcione offline
    blocos = [
        figura.to_html(
            full_html=False,
            include_plotlyjs='inline' if i == 0 else False,
            config={'responsive': True, 'displaylogo': False},
        )
        for i, figura in enumerate(figuras)
    ]

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard — Previsão de Risco de AVC</title>
<style>
  :root {{
    color-scheme: light;
    --superficie: {SUPERFICIE};
    --plano: #f9f9f7;
    --tinta: #0b0b0b;
    --tinta-secundaria: {TINTA_SECUNDARIA};
    --tinta-apagada: {TINTA_APAGADA};
    --borda: rgba(11, 11, 11, 0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 32px 24px 64px;
    background: var(--plano);
    color: var(--tinta);
    font-family: {FONTE};
  }}
  .conteudo {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .subtitulo {{ color: var(--tinta-secundaria); margin: 0 0 28px; font-size: 14px; }}
  .cartoes {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin-bottom: 28px;
  }}
  .cartao {{
    background: var(--superficie);
    border: 1px solid var(--borda);
    border-radius: 10px;
    padding: 18px 20px;
  }}
  .cartao-rotulo {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--tinta-apagada);
  }}
  .cartao-valor {{ font-size: 34px; font-weight: 600; margin: 6px 0 4px; }}
  .cartao-nota {{ font-size: 12px; color: var(--tinta-secundaria); line-height: 1.45; }}
  .grade {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 18px;
  }}
  .painel {{
    background: var(--superficie);
    border: 1px solid var(--borda);
    border-radius: 10px;
    padding: 8px;
    overflow-x: auto;
  }}
  .rodape {{
    margin-top: 32px;
    font-size: 12px;
    color: var(--tinta-apagada);
    line-height: 1.6;
    border-top: 1px solid var(--borda);
    padding-top: 16px;
  }}
  @media (max-width: 560px) {{
    .grade {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="conteudo">
  <h1>Previsão de Risco de AVC</h1>
  <p class="subtitulo">
    Modelo final: <strong>{metricas['nome_modelo']}</strong> ·
    limiar de decisão {metricas['limiar']:.3f} ·
    métricas medidas no conjunto de teste (1.022 pacientes nunca vistos no treino)
  </p>

  <div class="cartoes">
    {montar_cartoes(metricas)}
  </div>

  <div class="grade">
    {''.join(f'<div class="painel">{bloco}</div>' for bloco in blocos)}
  </div>

  <p class="rodape">
    Fonte: Healthcare Dataset Stroke Data (Kaggle / fedesoriano), 5.109 pacientes após a limpeza.<br>
    Gerado por <code>src/dashboard.py</code>. Projeto acadêmico de Métodos Quantitativos —
    <strong>não deve ser usado para decisão médica real</strong>.
  </p>
</div>
<script>
  // O primeiro gráfico é desenhado antes de a grade CSS existir, então ele mede um
  // container largo demais e teria as últimas barras cortadas para fora do painel.
  // Um redimensionamento após o `load` alinha todos os gráficos ao painel que os contém
  // (só aqui os desenhos já existem, porque `Plotly.newPlot` é assíncrono).
  // A partir daí, o `responsive: true` cuida das mudanças de tamanho da janela.
  window.addEventListener('load', () => {{
    document.querySelectorAll('.plotly-graph-div').forEach(g => Plotly.Plots.resize(g));
  }});
</script>
</body>
</html>
"""


def main():
    if not CAMINHO_METRICAS.exists() or not CAMINHO_MODELO.exists():
        raise SystemExit('Modelo ou métricas não encontrados. Rode antes: python src/train.py')

    print('1. Carregando dados, modelo e métricas...')
    df = carregar_e_limpar_dados(CAMINHO_DADOS)
    X, y = separar_features_e_alvo(df)
    metricas = json.loads(CAMINHO_METRICAS.read_text(encoding='utf-8'))
    modelo_salvo = joblib.load(CAMINHO_MODELO)

    print('2. Gerando os gráficos...')
    figuras = [
        grafico_desbalanceamento(y),
        grafico_risco_por_idade(df),
        grafico_efeito_smote(X, y),
        grafico_comparacao_modelos(metricas),
        grafico_curva_roc(metricas),
        grafico_matriz_confusao(metricas),
    ]

    grafico_fatores = grafico_fatores_de_risco(modelo_salvo)
    if grafico_fatores is not None:
        figuras.append(grafico_fatores)

    print('3. Montando o dashboard...')
    CAMINHO_DASHBOARD.write_text(montar_html(figuras, metricas), encoding='utf-8')
    print(f'   Dashboard: {CAMINHO_DASHBOARD.relative_to(DIRETORIO_RAIZ)}')


if __name__ == '__main__':
    main()
