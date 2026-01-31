"""
Módulo de Preprocessamento de Dados
Hackathon Participa DF - Categoria Acesso à Informação

Este módulo contém funções para carregar, limpar e preparar
os dados de pedidos de acesso à informação.
"""

import pandas as pd
import re
import unicodedata
from typing import List, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def carregar_dados(caminho_arquivo: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Carrega dados de um arquivo CSV.

    Args:
        caminho_arquivo: Caminho para o arquivo CSV
        encoding: Codificação do arquivo (padrão: utf-8)

    Returns:
        DataFrame com os dados carregados

    Raises:
        FileNotFoundError: Se o arquivo não existir
        pd.errors.EmptyDataError: Se o arquivo estiver vazio
    """
    try:
        logger.info(f"Carregando dados de: {caminho_arquivo}")
        df = pd.read_csv(caminho_arquivo, encoding=encoding)
        logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")

        # Validar colunas obrigatórias
        colunas_obrigatorias = ['id_pedido', 'texto_pedido']
        colunas_faltantes = [col for col in colunas_obrigatorias if col not in df.columns]

        if colunas_faltantes:
            raise ValueError(f"Colunas obrigatórias ausentes: {colunas_faltantes}")

        return df

    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {caminho_arquivo}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Arquivo vazio: {caminho_arquivo}")
        raise
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise


def remover_acentos(texto: str) -> str:
    """
    Remove acentos de caracteres, preservando o texto original.
    Útil para normalização em alguns casos de regex.

    Args:
        texto: Texto com acentos

    Returns:
        Texto sem acentos
    """
    if pd.isna(texto):
        return ""

    nfkd = unicodedata.normalize('NFKD', texto)
    texto_sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return texto_sem_acento


def limpar_texto(texto: str, remover_pontuacao: bool = False) -> str:
    """
    Realiza limpeza básica no texto.

    Args:
        texto: Texto a ser limpo
        remover_pontuacao: Se True, remove pontuação (padrão: False)

    Returns:
        Texto limpo
    """
    if pd.isna(texto):
        return ""

    # Converter para string
    texto = str(texto)

    # Remover espaços múltiplos
    texto = re.sub(r'\s+', ' ', texto)

    # Remover espaços no início e fim
    texto = texto.strip()

    # Opcional: remover pontuação (cuidado: pode afetar detecção de CPF, email, etc.)
    if remover_pontuacao:
        # Preservar hífen, ponto, @, etc. importantes para dados pessoais
        # Remover apenas pontuação claramente irrelevante
        texto = re.sub(r'[^\w\s@.\-\(\)\+]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


def normalizar_numeros(texto: str) -> str:
    """
    Normaliza formatos de números (CPF, telefone, etc.) para facilitar detecção.
    IMPORTANTE: Esta função é para análise, não para substituir o texto original.

    Args:
        texto: Texto original

    Returns:
        Versão com números normalizados
    """
    if pd.isna(texto):
        return ""

    texto = str(texto)

    # Preservar o texto original, apenas adicionar versão normalizada
    # (Pode ser útil criar feature adicional)

    return texto


def validar_qualidade_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida e limpa dados com problemas de qualidade.

    Args:
        df: DataFrame original

    Returns:
        DataFrame validado e limpo
    """
    logger.info("Validando qualidade dos dados...")

    df_limpo = df.copy()

    # Remover duplicatas baseado em id_pedido
    duplicatas_antes = len(df_limpo)
    df_limpo = df_limpo.drop_duplicates(subset=['id_pedido'], keep='first')
    duplicatas_removidas = duplicatas_antes - len(df_limpo)
    if duplicatas_removidas > 0:
        logger.warning(f"Removidas {duplicatas_removidas} linhas duplicadas")

    # Remover linhas com texto_pedido vazio ou nulo
    vazios_antes = len(df_limpo)
    df_limpo = df_limpo[df_limpo['texto_pedido'].notna()]
    df_limpo = df_limpo[df_limpo['texto_pedido'].astype(str).str.strip() != '']
    vazios_removidos = vazios_antes - len(df_limpo)
    if vazios_removidos > 0:
        logger.warning(f"Removidas {vazios_removidos} linhas com texto vazio")

    # Resetar índice
    df_limpo = df_limpo.reset_index(drop=True)

    logger.info(f"Dados validados. Shape final: {df_limpo.shape}")

    return df_limpo


def preprocessar_dados(
    df: pd.DataFrame,
    coluna_texto: str = 'texto_pedido',
    criar_features: bool = True
) -> pd.DataFrame:
    """
    Aplica preprocessamento completo nos dados.

    Args:
        df: DataFrame com os dados
        coluna_texto: Nome da coluna com o texto dos pedidos
        criar_features: Se True, cria features adicionais para ML

    Returns:
        DataFrame preprocessado
    """
    logger.info("Iniciando preprocessamento dos dados...")

    df_proc = df.copy()

    # Limpar texto
    logger.info("Limpando textos...")
    df_proc['texto_limpo'] = df_proc[coluna_texto].apply(limpar_texto)

    # Criar features adicionais para ML
    if criar_features:
        logger.info("Criando features adicionais...")

        # Feature: tamanho do texto
        df_proc['tamanho_texto'] = df_proc['texto_limpo'].apply(len)

        # Feature: quantidade de números no texto
        df_proc['qtd_numeros'] = df_proc['texto_limpo'].apply(
            lambda x: len(re.findall(r'\d', str(x)))
        )

        # Feature: quantidade de palavras
        df_proc['qtd_palavras'] = df_proc['texto_limpo'].apply(
            lambda x: len(str(x).split())
        )

        # Feature: tem @ (possível email)
        df_proc['tem_arroba'] = df_proc['texto_limpo'].apply(
            lambda x: '@' in str(x)
        )

        # Feature: sequências longas de números (possível CPF, telefone)
        df_proc['tem_seq_numeros_longa'] = df_proc['texto_limpo'].apply(
            lambda x: bool(re.search(r'\d{9,}', str(x)))
        )

        # Feature: razão números/caracteres
        df_proc['razao_numeros'] = df_proc.apply(
            lambda row: row['qtd_numeros'] / max(row['tamanho_texto'], 1),
            axis=1
        )

    logger.info("Preprocessamento concluído!")

    return df_proc


def split_treino_validacao(
    df: pd.DataFrame,
    coluna_label: str = 'contem_dados_pessoais',
    validacao_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Divide dados em treino e validação de forma estratificada.

    Args:
        df: DataFrame com os dados
        coluna_label: Nome da coluna com os rótulos
        validacao_size: Proporção dos dados para validação
        random_state: Seed para reprodutibilidade

    Returns:
        Tupla (df_treino, df_validacao)
    """
    from sklearn.model_selection import train_test_split

    logger.info(f"Dividindo dados em treino/validação (validação={validacao_size*100}%)...")

    if coluna_label not in df.columns:
        raise ValueError(f"Coluna de label '{coluna_label}' não encontrada nos dados")

    # Split estratificado para manter proporção das classes
    df_treino, df_validacao = train_test_split(
        df,
        test_size=validacao_size,
        stratify=df[coluna_label],
        random_state=random_state
    )

    logger.info(f"Treino: {len(df_treino)} amostras")
    logger.info(f"Validação: {len(df_validacao)} amostras")

    # Mostrar distribuição das classes
    logger.info("Distribuição no treino:")
    logger.info(df_treino[coluna_label].value_counts())
    logger.info("Distribuição na validação:")
    logger.info(df_validacao[coluna_label].value_counts())

    return df_treino, df_validacao


def salvar_dados_preprocessados(
    df: pd.DataFrame,
    caminho_saida: str,
    index: bool = False
) -> None:
    """
    Salva dados preprocessados em arquivo CSV.

    Args:
        df: DataFrame a ser salvo
        caminho_saida: Caminho do arquivo de saída
        index: Se True, salva o índice do DataFrame
    """
    try:
        logger.info(f"Salvando dados preprocessados em: {caminho_saida}")
        df.to_csv(caminho_saida, index=index, encoding='utf-8')
        logger.info("Dados salvos com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao salvar dados: {str(e)}")
        raise


# Função de exemplo/teste
if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de preprocessamento carregado com sucesso!")
    print("\nFunções disponíveis:")
    print("- carregar_dados()")
    print("- limpar_texto()")
    print("- preprocessar_dados()")
    print("- split_treino_validacao()")
    print("- salvar_dados_preprocessados()")

    # Teste básico
    texto_exemplo = "  Olá,  gostaria de informações sobre   o processo   123.456.789-00  "
    print(f"\nTeste de limpeza:")
    print(f"Antes: '{texto_exemplo}'")
    print(f"Depois: '{limpar_texto(texto_exemplo)}'")
