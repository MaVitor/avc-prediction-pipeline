"""
Camada que conversa com o modelo treinado na Fase 3.

É o único ponto do projeto que carrega o .pkl e chama o `predict_proba`. Tanto a API
JSON (`apis.py`) quanto o front-end HTMX (`views.py`) passam por aqui, de modo que a
regra de decisão exista uma vez só e as duas interfaces nunca divirjam.
"""

from functools import lru_cache

import joblib
import pandas as pd
from django.conf import settings


class ModeloIndisponivel(RuntimeError):
    """Levantado quando o .pkl ainda não foi gerado."""


@lru_cache(maxsize=1)
def carregar_modelo():
    """
    Lê o modelo do disco uma única vez e mantém em memória.

    Sem o cache, cada requisição pagaria o custo de desserializar o pipeline inteiro.
    """
    if not settings.CAMINHO_MODELO.exists():
        raise ModeloIndisponivel(
            f'Modelo não encontrado em {settings.CAMINHO_MODELO}. '
            'Rode antes: python src/train.py'
        )

    return joblib.load(settings.CAMINHO_MODELO)


def classificar_faixa(probabilidade, limiar):
    """
    Traduz a probabilidade em uma faixa de risco legível.

    O limiar veio da validação e é o corte que decide "alto risco". A faixa intermediária
    (metade do limiar até ele) existe só para a leitura humana: separa quem está claramente
    fora de risco de quem está perto da linha de corte.
    """
    if probabilidade >= limiar:
        return 'Alto'
    if probabilidade >= limiar / 2:
        return 'Moderado'

    return 'Baixo'


def prever(dados_paciente):
    """
    Recebe os dados brutos de um paciente (dict) e devolve a previsão de risco.

    Não faz imputação, escalonamento nem encoding: tudo isso já está dentro do
    pipeline exportado, então o dado entra exatamente no formato do CSV original.
    O `bmi` pode vir como None — o imputador do pipeline preenche com a mediana do treino.
    """
    modelo_salvo = carregar_modelo()
    pipeline = modelo_salvo['pipeline']
    limiar = modelo_salvo['limiar']

    # O pipeline foi treinado com um DataFrame, então espera as colunas na mesma ordem
    entrada = pd.DataFrame([dados_paciente], columns=modelo_salvo['colunas'])

    probabilidade = float(pipeline.predict_proba(entrada)[0, 1])

    return {
        'probabilidade': probabilidade,
        'percentual': round(probabilidade * 100, 1),
        'alto_risco': probabilidade >= limiar,
        'faixa_risco': classificar_faixa(probabilidade, limiar),
        'limiar': limiar,
        'limiar_percentual': round(limiar * 100, 1),
        'modelo': modelo_salvo['nome_modelo'],
        'imc_imputado': dados_paciente.get('bmi') is None,
    }
