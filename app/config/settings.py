"""
Configuração mínima do Django para servir o modelo de AVC.

O projeto não tem banco de dados nem autenticação de propósito: a aplicação apenas
carrega um arquivo .pkl e responde previsões. Manter o Django enxuto evita migrations
e um `db.sqlite3` que não teriam utilidade nenhuma aqui.
"""

import os
from pathlib import Path

# BASE_DIR aponta para a pasta `app/`; o modelo e os relatórios ficam na raiz do repositório
BASE_DIR = Path(__file__).resolve().parent.parent
DIRETORIO_RAIZ = BASE_DIR.parent

CAMINHO_MODELO = DIRETORIO_RAIZ / 'models' / 'modelo_avc.pkl'
CAMINHO_DASHBOARD = DIRETORIO_RAIZ / 'reports' / 'dashboard.html'

# Chave de desenvolvimento. Em um deploy real, defina DJANGO_SECRET_KEY no ambiente.
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'chave-insegura-apenas-para-desenvolvimento')

DEBUG = os.environ.get('DJANGO_DEBUG', '1') == '1'

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '[::1]']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'predicao',
]

MIDDLEWARE = [
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Sem banco: nenhuma parte da aplicação persiste dados
DATABASES = {}

LANGUAGE_CODE = 'pt-br'
TIME_ZONE = 'America/Sao_Paulo'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
