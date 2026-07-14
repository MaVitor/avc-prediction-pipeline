"""
Rotas do projeto.

    /            -> formulário HTMX (front-end)
    /prever      -> fragmento HTML devolvido ao HTMX
    /dashboard   -> painel da Fase 4
    /api/prever  -> API JSON (Django Ninja)
    /api/docs    -> documentação interativa gerada automaticamente
"""

from django.urls import path
from ninja import NinjaAPI

from predicao import views
from predicao.apis import router as router_predicao

api = NinjaAPI(
    title='API de Previsão de Risco de AVC',
    version='1.0.0',
    description=(
        'Serve o modelo de Machine Learning treinado na Fase 3. '
        'Projeto acadêmico — não use para decisão médica real.'
    ),
)
api.add_router('', router_predicao)

urlpatterns = [
    path('', views.formulario, name='formulario'),
    path('prever', views.prever_html, name='prever_html'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('api/', api.urls),
]
