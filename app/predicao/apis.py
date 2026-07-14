"""
API JSON que serve o modelo (Django Ninja).

Documentação interativa gerada automaticamente a partir dos tipos: /api/docs
"""

from ninja import Router

from .schemas import ErroSchema, PacienteSchema, PrevisaoSchema
from .servico import ModeloIndisponivel, carregar_modelo, prever

router = Router(tags=['Previsão de AVC'])


@router.post(
    '/prever',
    response={200: PrevisaoSchema, 503: ErroSchema},
    summary='Prevê o risco de AVC de um paciente',
)
def prever_risco(request, paciente: PacienteSchema):
    """
    Recebe os dados brutos do paciente e devolve a probabilidade de AVC.

    O `bmi` pode ser omitido: o pipeline imputa a mediana do treino, do mesmo jeito
    que fez com os 201 registros sem IMC na base original.
    """
    try:
        return 200, prever(paciente.dict())
    except ModeloIndisponivel as erro:
        return 503, {'detalhe': str(erro)}


@router.get('/saude', summary='Verifica se a API e o modelo estão no ar')
def saude(request):
    """
    Endpoint de health check: confirma que o .pkl foi carregado com sucesso.
    """
    try:
        modelo_salvo = carregar_modelo()
    except ModeloIndisponivel as erro:
        return 503, {'status': 'sem modelo', 'detalhe': str(erro)}

    return {
        'status': 'ok',
        'modelo': modelo_salvo['nome_modelo'],
        'limiar': modelo_salvo['limiar'],
    }
