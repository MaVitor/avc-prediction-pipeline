"""
Contratos de entrada e saída da API (Django Ninja + Pydantic).

A tipagem aqui é o que gera a validação automática e a documentação em /api/docs.
Os nomes dos campos são iguais aos do CSV original de propósito: é o formato que o
pipeline treinado espera receber.
"""

from typing import Literal, Optional

from ninja import Schema
from pydantic import Field


class PacienteSchema(Schema):
    """Dados de um paciente, no mesmo formato do dataset bruto."""

    gender: Literal['Male', 'Female'] = Field(description='Gênero do paciente')
    age: float = Field(ge=0, le=120, description='Idade em anos')
    hypertension: Literal[0, 1] = Field(description='1 se tem hipertensão')
    heart_disease: Literal[0, 1] = Field(description='1 se tem doença cardíaca')
    ever_married: Literal['Yes', 'No'] = Field(description='Já foi casado(a)')
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'] = (
        Field(description='Tipo de ocupação')
    )
    Residence_type: Literal['Urban', 'Rural'] = Field(description='Tipo de residência')
    avg_glucose_level: float = Field(ge=0, le=500, description='Nível médio de glicose (mg/dL)')
    # Opcional de propósito: é a coluna com 201 nulos na base original. Quando vem vazia,
    # o imputador do pipeline preenche com a mediana aprendida no treino.
    bmi: Optional[float] = Field(default=None, ge=0, le=100, description='IMC (opcional)')
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes', 'Unknown'] = Field(
        description='Situação de tabagismo'
    )

    class Config:
        json_schema_extra = {
            'example': {
                'gender': 'Male',
                'age': 67,
                'hypertension': 0,
                'heart_disease': 1,
                'ever_married': 'Yes',
                'work_type': 'Private',
                'Residence_type': 'Urban',
                'avg_glucose_level': 228.69,
                'bmi': 36.6,
                'smoking_status': 'formerly smoked',
            }
        }


class PrevisaoSchema(Schema):
    """Resposta da previsão."""

    probabilidade: float = Field(description='Probabilidade de AVC, de 0 a 1')
    percentual: float = Field(description='A mesma probabilidade, em porcentagem')
    alto_risco: bool = Field(description='True quando a probabilidade atinge o limiar')
    faixa_risco: Literal['Baixo', 'Moderado', 'Alto'] = Field(description='Faixa de leitura')
    limiar: float = Field(description='Limiar de decisão definido na validação')
    limiar_percentual: float = Field(description='O mesmo limiar, em porcentagem')
    modelo: str = Field(description='Algoritmo que gerou a previsão')
    imc_imputado: bool = Field(description='True se o IMC não foi informado e foi imputado')


class ErroSchema(Schema):
    detalhe: str
