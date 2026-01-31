"""
Sistema de Detecção de Dados Pessoais em Pedidos de Acesso à Informação
Hackathon Participa DF - Categoria Acesso à Informação

Este pacote implementa uma solução híbrida para identificar pedidos que
contenham dados pessoais (nome, CPF, RG, telefone, e-mail).
"""

__version__ = "1.0.0"
__author__ = "Henrique Lacerda Silveira"

from .modelo import ModeloHibridoDeteccao, DetectorPadroesRegex, DetectorNER
from .preprocessamento import carregar_dados, preprocessar_dados
from .avaliacao import avaliar_modelo, calcular_f1_score

__all__ = [
    'ModeloHibridoDeteccao',
    'DetectorPadroesRegex',
    'DetectorNER',
    'carregar_dados',
    'preprocessar_dados',
    'avaliar_modelo',
    'calcular_f1_score'
]
