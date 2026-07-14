"""
Front-end simples: um formulário que fala com o servidor via HTMX.

O HTMX espera receber HTML, e não JSON, para trocar um pedaço da página. Por isso a
`prever_html` devolve um fragmento renderizado em vez de chamar `/api/prever`. As duas
rotas chamam o mesmo `servico.prever()`, então a regra de decisão continua existindo
em um lugar só — o que muda é apenas o formato da resposta.
"""

from django.http import Http404, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from pydantic import ValidationError

from .schemas import PacienteSchema
from .servico import ModeloIndisponivel, prever

# Usado para transformar o erro técnico do Pydantic no nome do campo que a pessoa vê na tela
ROTULOS_CAMPOS = {
    'gender': 'Gênero',
    'age': 'Idade',
    'hypertension': 'Hipertensão',
    'heart_disease': 'Doença cardíaca',
    'ever_married': 'Já foi casado(a)',
    'work_type': 'Ocupação',
    'Residence_type': 'Residência',
    'avg_glucose_level': 'Glicose média',
    'bmi': 'IMC',
    'smoking_status': 'Tabagismo',
}

# Opções dos campos de seleção: rótulo em português, valor no formato que o modelo espera
OPCOES = {
    'gender': [('Male', 'Masculino'), ('Female', 'Feminino')],
    'ever_married': [('Yes', 'Sim'), ('No', 'Não')],
    'Residence_type': [('Urban', 'Urbana'), ('Rural', 'Rural')],
    'work_type': [
        ('Private', 'Setor privado'),
        ('Self-employed', 'Autônomo'),
        ('Govt_job', 'Setor público'),
        ('children', 'Criança'),
        ('Never_worked', 'Nunca trabalhou'),
    ],
    'smoking_status': [
        ('never smoked', 'Nunca fumou'),
        ('formerly smoked', 'Ex-fumante'),
        ('smokes', 'Fuma'),
        ('Unknown', 'Não informado'),
    ],
    'sim_nao': [(0, 'Não'), (1, 'Sim')],
}


@require_GET
def formulario(request):
    """Página inicial com o formulário de avaliação."""
    return render(request, 'predicao/index.html', {'opcoes': OPCOES})


@require_POST
def prever_html(request):
    """
    Recebe o formulário via HTMX e devolve o cartão de resultado como fragmento HTML.
    """
    dados = request.POST.dict()

    # Campo vazio significa "IMC não informado", que é justamente o caso que o
    # imputador do pipeline resolve. String vazia viraria erro de validação.
    if not dados.get('bmi'):
        dados['bmi'] = None

    # Um formulário HTML envia tudo como texto, e `Literal[0, 1]` não aceita a string '0'.
    # A API JSON não precisa disso porque JSON já distingue número de texto.
    for campo in ('hypertension', 'heart_disease'):
        if campo in dados:
            try:
                dados[campo] = int(dados[campo])
            except (TypeError, ValueError):
                pass  # deixa o schema reportar o valor inválido

    try:
        # Reaproveita o schema da API para validar: uma única definição de regra
        paciente = PacienteSchema(**dados)
    except ValidationError as erro:
        # O texto cru do Pydantic é técnico e em inglês; a tela mostra só os campos a corrigir
        campos = {
            ROTULOS_CAMPOS.get(str(detalhe['loc'][0]), str(detalhe['loc'][0]))
            for detalhe in erro.errors()
            if detalhe['loc']
        }
        return render(
            request,
            'predicao/resultado.html',
            {'erro': f'Verifique os campos: {", ".join(sorted(campos))}.'},
            status=400,
        )

    try:
        resultado = prever(paciente.dict())
    except ModeloIndisponivel as erro:
        return render(request, 'predicao/resultado.html', {'erro': str(erro)}, status=503)

    return render(request, 'predicao/resultado.html', {'resultado': resultado})


@require_GET
def dashboard(request):
    """
    Serve o painel estático gerado na Fase 4 por `src/dashboard.py`.
    """
    from django.conf import settings

    if not settings.CAMINHO_DASHBOARD.exists():
        raise Http404('Dashboard ainda não foi gerado. Rode: python src/dashboard.py')

    return HttpResponse(settings.CAMINHO_DASHBOARD.read_text(encoding='utf-8'))
