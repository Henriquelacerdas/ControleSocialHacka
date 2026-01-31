"""
Módulo de Avaliação de Modelos
Hackathon Participa DF - Categoria Acesso à Informação

Este módulo implementa as métricas de avaliação conforme
especificado no regulamento do hackathon:
- Precisão (Precision)
- Sensibilidade/Recall
- F1-Score
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calcular_matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Calcula a matriz de confusão.

    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos

    Returns:
        Dicionário com VP, VN, FP, FN
    """
    y_true = np.array(y_true).astype(bool)
    y_pred = np.array(y_pred).astype(bool)

    vp = np.sum((y_true == True) & (y_pred == True))  # Verdadeiros Positivos
    vn = np.sum((y_true == False) & (y_pred == False))  # Verdadeiros Negativos
    fp = np.sum((y_true == False) & (y_pred == True))  # Falsos Positivos
    fn = np.sum((y_true == True) & (y_pred == False))  # Falsos Negativos

    return {
        'VP': int(vp),
        'VN': int(vn),
        'FP': int(fp),
        'FN': int(fn)
    }


def calcular_precisao(vp: int, fp: int) -> float:
    """
    Calcula a Precisão (Precision).

    Precisão = VP / (VP + FP)

    Args:
        vp: Verdadeiros Positivos
        fp: Falsos Positivos

    Returns:
        Precisão (entre 0 e 1)
    """
    if (vp + fp) == 0:
        return 0.0
    return vp / (vp + fp)


def calcular_recall(vp: int, fn: int) -> float:
    """
    Calcula o Recall/Sensibilidade.

    Recall = VP / (VP + FN)

    Args:
        vp: Verdadeiros Positivos
        fn: Falsos Negativos

    Returns:
        Recall (entre 0 e 1)
    """
    if (vp + fn) == 0:
        return 0.0
    return vp / (vp + fn)


def calcular_f1_score(precisao: float, recall: float) -> float:
    """
    Calcula o F1-Score.

    F1-Score = 2 × (Precisão × Recall) / (Precisão + Recall)

    Args:
        precisao: Precisão
        recall: Recall

    Returns:
        F1-Score (entre 0 e 1)
    """
    if (precisao + recall) == 0:
        return 0.0
    return 2 * (precisao * recall) / (precisao + recall)


def calcular_acuracia(vp: int, vn: int, fp: int, fn: int) -> float:
    """
    Calcula a Acurácia.

    Acurácia = (VP + VN) / (VP + VN + FP + FN)

    Args:
        vp: Verdadeiros Positivos
        vn: Verdadeiros Negativos
        fp: Falsos Positivos
        fn: Falsos Negativos

    Returns:
        Acurácia (entre 0 e 1)
    """
    total = vp + vn + fp + fn
    if total == 0:
        return 0.0
    return (vp + vn) / total


def avaliar_modelo(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mostrar_detalhes: bool = True
) -> Dict[str, float]:
    """
    Avalia o modelo usando as métricas do hackathon.

    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos
        mostrar_detalhes: Se True, imprime detalhes da avaliação

    Returns:
        Dicionário com todas as métricas
    """
    # Calcular matriz de confusão
    matriz = calcular_matriz_confusao(y_true, y_pred)

    vp = matriz['VP']
    vn = matriz['VN']
    fp = matriz['FP']
    fn = matriz['FN']

    # Calcular métricas
    precisao = calcular_precisao(vp, fp)
    recall = calcular_recall(vp, fn)
    f1 = calcular_f1_score(precisao, recall)
    acuracia = calcular_acuracia(vp, vn, fp, fn)

    metricas = {
        'VP': vp,
        'VN': vn,
        'FP': fp,
        'FN': fn,
        'Precisao': precisao,
        'Recall': recall,
        'F1-Score': f1,
        'Acuracia': acuracia
    }

    if mostrar_detalhes:
        logger.info("\n" + "="*60)
        logger.info("RELATÓRIO DE AVALIAÇÃO DO MODELO")
        logger.info("="*60)
        logger.info("\nMatriz de Confusão:")
        logger.info(f"  Verdadeiros Positivos (VP): {vp}")
        logger.info(f"  Verdadeiros Negativos (VN): {vn}")
        logger.info(f"  Falsos Positivos (FP):      {fp}")
        logger.info(f"  Falsos Negativos (FN):      {fn}")
        logger.info("\nMétricas de Desempenho:")
        logger.info(f"  Precisão:   {precisao:.4f} ({precisao*100:.2f}%)")
        logger.info(f"  Recall:     {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"  F1-Score:   {f1:.4f} ({f1*100:.2f}%)")
        logger.info(f"  Acurácia:   {acuracia:.4f} ({acuracia*100:.2f}%)")
        logger.info("\nInterpretação para o Hackathon:")
        logger.info(f"  Pontuação P1 (F1-Score): {f1:.4f}")
        logger.info(f"  Em caso de empate:")
        logger.info(f"    - Falsos Negativos: {fn} (menor é melhor)")
        logger.info(f"    - Falsos Positivos: {fp} (menor é melhor)")
        logger.info("="*60 + "\n")

    return metricas


def plotar_matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    salvar_caminho: Optional[str] = None
):
    """
    Plota a matriz de confusão de forma visual.

    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos
        salvar_caminho: Caminho para salvar o gráfico (opcional)
    """
    from sklearn.metrics import confusion_matrix

    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Plotar
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Não contém', 'Contém dados'],
        yticklabels=['Não contém', 'Contém dados']
    )
    plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Valor Predito', fontsize=12)
    plt.tight_layout()

    if salvar_caminho:
        Path(salvar_caminho).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(salvar_caminho, dpi=300, bbox_inches='tight')
        logger.info(f"Matriz de confusão salva em: {salvar_caminho}")

    plt.show()


def analisar_erros(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coluna_texto: str = 'texto_pedido',
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Analisa os erros do modelo (falsos positivos e falsos negativos).

    Args:
        df: DataFrame original com os textos
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos
        coluna_texto: Nome da coluna com o texto
        top_n: Número de exemplos a retornar

    Returns:
        Dicionário com DataFrames de falsos positivos e falsos negativos
    """
    df_analise = df.copy()
    df_analise['y_true'] = y_true
    df_analise['y_pred'] = y_pred

    # Identificar erros
    df_analise['erro_tipo'] = 'Correto'
    df_analise.loc[
        (df_analise['y_true'] == False) & (df_analise['y_pred'] == True),
        'erro_tipo'
    ] = 'Falso Positivo'
    df_analise.loc[
        (df_analise['y_true'] == True) & (df_analise['y_pred'] == False),
        'erro_tipo'
    ] = 'Falso Negativo'

    # Extrair erros
    falsos_positivos = df_analise[df_analise['erro_tipo'] == 'Falso Positivo'].head(top_n)
    falsos_negativos = df_analise[df_analise['erro_tipo'] == 'Falso Negativo'].head(top_n)

    logger.info(f"\nTotal de Falsos Positivos: {len(df_analise[df_analise['erro_tipo'] == 'Falso Positivo'])}")
    logger.info(f"Total de Falsos Negativos: {len(df_analise[df_analise['erro_tipo'] == 'Falso Negativo'])}")

    if len(falsos_positivos) > 0:
        logger.info(f"\nExemplos de Falsos Positivos (top {top_n}):")
        for idx, row in falsos_positivos.iterrows():
            texto = row[coluna_texto][:200]  # Primeiros 200 caracteres
            logger.info(f"  - {texto}...")

    if len(falsos_negativos) > 0:
        logger.info(f"\nExemplos de Falsos Negativos (top {top_n}):")
        for idx, row in falsos_negativos.iterrows():
            texto = row[coluna_texto][:200]
            logger.info(f"  - {texto}...")

    return {
        'falsos_positivos': falsos_positivos,
        'falsos_negativos': falsos_negativos
    }


def salvar_relatorio(
    metricas: Dict[str, float],
    caminho_saida: str,
    adicionar_info: Optional[Dict] = None
):
    """
    Salva relatório de avaliação em arquivo JSON.

    Args:
        metricas: Dicionário com métricas
        caminho_saida: Caminho do arquivo de saída
        adicionar_info: Informações adicionais a incluir
    """
    import json

    relatorio = {
        'metricas': metricas,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    if adicionar_info:
        relatorio.update(adicionar_info)

    # Converter tipos numpy para tipos Python nativos
    relatorio_serializable = {}
    for key, value in relatorio.items():
        if isinstance(value, dict):
            relatorio_serializable[key] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v
                for k, v in value.items()
            }
        else:
            relatorio_serializable[key] = value

    # Criar diretório se não existir
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)

    # Salvar
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(relatorio_serializable, f, indent=2, ensure_ascii=False)

    logger.info(f"Relatório salvo em: {caminho_saida}")


def comparar_modelos(
    resultados_modelos: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Compara resultados de múltiplos modelos.

    Args:
        resultados_modelos: Dicionário com nome do modelo e suas métricas

    Returns:
        DataFrame com comparação dos modelos
    """
    df_comparacao = pd.DataFrame(resultados_modelos).T

    # Ordenar por F1-Score (principal métrica do hackathon)
    df_comparacao = df_comparacao.sort_values('F1-Score', ascending=False)

    logger.info("\n" + "="*60)
    logger.info("COMPARAÇÃO DE MODELOS")
    logger.info("="*60)
    logger.info(df_comparacao.to_string())
    logger.info("="*60 + "\n")

    return df_comparacao


def validacao_cruzada_custom(
    modelo,
    df: pd.DataFrame,
    coluna_texto: str = 'texto_pedido',
    coluna_label: str = 'contem_dados_pessoais',
    n_folds: int = 5
) -> Dict[str, List[float]]:
    """
    Realiza validação cruzada customizada.

    Args:
        modelo: Instância do modelo
        df: DataFrame com os dados
        coluna_texto: Nome da coluna com texto
        coluna_label: Nome da coluna com labels
        n_folds: Número de folds

    Returns:
        Dicionário com listas de métricas por fold
    """
    from sklearn.model_selection import StratifiedKFold

    logger.info(f"\nIniciando validação cruzada com {n_folds} folds...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    resultados = {
        'Precisao': [],
        'Recall': [],
        'F1-Score': [],
        'VP': [],
        'FP': [],
        'FN': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[coluna_label]), 1):
        logger.info(f"\nFold {fold}/{n_folds}")

        # Dividir dados
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        # Treinar
        modelo.treinar(df_train, coluna_texto=coluna_texto, coluna_label=coluna_label)

        # Prever
        y_val_true = df_val[coluna_label].values
        y_val_pred = modelo.prever_batch(df_val[coluna_texto])

        # Avaliar
        metricas = avaliar_modelo(y_val_true, y_val_pred, mostrar_detalhes=False)

        # Armazenar
        resultados['Precisao'].append(metricas['Precisao'])
        resultados['Recall'].append(metricas['Recall'])
        resultados['F1-Score'].append(metricas['F1-Score'])
        resultados['VP'].append(metricas['VP'])
        resultados['FP'].append(metricas['FP'])
        resultados['FN'].append(metricas['FN'])

        logger.info(f"  F1-Score: {metricas['F1-Score']:.4f}")

    # Calcular médias
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS DA VALIDAÇÃO CRUZADA")
    logger.info("="*60)
    for metrica, valores in resultados.items():
        media = np.mean(valores)
        std = np.std(valores)
        logger.info(f"{metrica}: {media:.4f} ± {std:.4f}")
    logger.info("="*60 + "\n")

    return resultados


# Exemplo de uso
if __name__ == "__main__":
    print("Módulo de avaliação carregado com sucesso!")

    # Teste com dados fictícios
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    print("\nTeste de avaliação:")
    metricas = avaliar_modelo(y_true, y_pred)
