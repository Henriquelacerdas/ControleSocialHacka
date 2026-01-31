#!/usr/bin/env python3
"""
Script Principal - Detecção de Dados Pessoais
Hackathon Participa DF - Categoria Acesso à Informação

Este script fornece interface de linha de comando para:
- Treinar o modelo
- Fazer predições
- Avaliar performance
"""

import argparse
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Adicionar diretório src ao path
sys.path.append(str(Path(__file__).parent))

from preprocessamento import (
    carregar_dados,
    validar_qualidade_dados,
    preprocessar_dados,
    split_treino_validacao
)
from modelo import ModeloHibridoDeteccao
from avaliacao import (
    avaliar_modelo,
    plotar_matriz_confusao,
    analisar_erros,
    salvar_relatorio
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def treinar_modelo(args):
    """
    Modo de treinamento do modelo.

    Args:
        args: Argumentos da linha de comando
    """
    logger.info("="*60)
    logger.info("MODO TREINAMENTO")
    logger.info("="*60)

    # Carregar dados
    logger.info(f"\n1. Carregando dados de: {args.data}")
    df = carregar_dados(args.data)

    # Validar qualidade
    logger.info("\n2. Validando qualidade dos dados...")
    df = validar_qualidade_dados(df)

    # Preprocessar
    logger.info("\n3. Preprocessando dados...")
    df = preprocessar_dados(df, coluna_texto='texto_pedido', criar_features=True)

    # Split treino/validação se solicitado
    if args.validacao:
        logger.info("\n4. Dividindo dados em treino/validação...")
        df_treino, df_validacao = split_treino_validacao(
            df,
            coluna_label='contem_dados_pessoais',
            validacao_size=0.2
        )
    else:
        df_treino = df
        df_validacao = None

    # Criar e treinar modelo
    logger.info("\n5. Criando e treinando modelo...")
    modelo = ModeloHibridoDeteccao(
        usar_regex=True,
        usar_ner=True,
        usar_ml=True,
        threshold_confianca=args.threshold
    )

    modelo.treinar(
        df_treino,
        coluna_texto='texto_limpo',
        coluna_label='contem_dados_pessoais'
    )

    # Avaliar em validação se disponível
    if df_validacao is not None:
        logger.info("\n6. Avaliando em conjunto de validação...")
        y_val_true = df_validacao['contem_dados_pessoais'].values
        y_val_pred = modelo.prever_batch(df_validacao['texto_limpo'])

        metricas = avaliar_modelo(y_val_true, y_val_pred, mostrar_detalhes=True)

        # Análise de erros
        if args.analisar_erros:
            logger.info("\n7. Analisando erros...")
            analisar_erros(
                df_validacao,
                y_val_true,
                y_val_pred,
                coluna_texto='texto_pedido',
                top_n=5
            )

        # Plotar matriz de confusão
        if args.plotar:
            logger.info("\n8. Plotando matriz de confusão...")
            caminho_plot = Path(args.output).parent / "matriz_confusao.png"
            plotar_matriz_confusao(y_val_true, y_val_pred, salvar_caminho=str(caminho_plot))

        # Salvar relatório
        if args.salvar_relatorio:
            logger.info("\n9. Salvando relatório...")
            caminho_relatorio = Path(args.output).parent / "relatorio_avaliacao.json"
            salvar_relatorio(
                metricas,
                str(caminho_relatorio),
                adicionar_info={
                    'modelo': 'ModeloHibridoDeteccao',
                    'threshold': args.threshold,
                    'dados_treino': str(args.data)
                }
            )

    # Salvar modelo
    logger.info(f"\n10. Salvando modelo em: {args.output}")
    modelo.salvar(args.output)

    logger.info("\n" + "="*60)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("="*60)


def prever(args):
    """
    Modo de predição.

    Args:
        args: Argumentos da linha de comando
    """
    logger.info("="*60)
    logger.info("MODO PREDIÇÃO")
    logger.info("="*60)

    # Carregar modelo
    logger.info(f"\n1. Carregando modelo de: {args.model}")
    modelo = ModeloHibridoDeteccao.carregar(args.model)

    # Carregar dados
    logger.info(f"\n2. Carregando dados de: {args.input}")
    df = carregar_dados(args.input)

    # Validar qualidade
    logger.info("\n3. Validando qualidade dos dados...")
    df = validar_qualidade_dados(df)

    # Preprocessar
    logger.info("\n4. Preprocessando dados...")
    df = preprocessar_dados(df, coluna_texto='texto_pedido', criar_features=True)

    # Fazer predições
    logger.info("\n5. Fazendo predições...")
    predicoes = modelo.prever_batch(df['texto_limpo'])

    # Preparar resultado
    resultado = pd.DataFrame({
        'id_pedido': df['id_pedido'],
        'contem_dados_pessoais': predicoes
    })

    # Salvar
    logger.info(f"\n6. Salvando predições em: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.output, index=False)

    # Estatísticas
    logger.info("\nEstatísticas das predições:")
    logger.info(f"  Total de pedidos analisados: {len(resultado)}")
    logger.info(f"  Pedidos COM dados pessoais: {resultado['contem_dados_pessoais'].sum()}")
    logger.info(f"  Pedidos SEM dados pessoais: {(~resultado['contem_dados_pessoais']).sum()}")
    logger.info(f"  Proporção com dados: {resultado['contem_dados_pessoais'].mean()*100:.2f}%")

    logger.info("\n" + "="*60)
    logger.info("PREDIÇÕES CONCLUÍDAS COM SUCESSO!")
    logger.info("="*60)


def avaliar(args):
    """
    Modo de avaliação.

    Args:
        args: Argumentos da linha de comando
    """
    logger.info("="*60)
    logger.info("MODO AVALIAÇÃO")
    logger.info("="*60)

    # Carregar modelo
    logger.info(f"\n1. Carregando modelo de: {args.model}")
    modelo = ModeloHibridoDeteccao.carregar(args.model)

    # Carregar dados
    logger.info(f"\n2. Carregando dados de: {args.data}")
    df = carregar_dados(args.data)

    # Validar coluna de labels
    if 'contem_dados_pessoais' not in df.columns:
        logger.error("ERRO: Dados de avaliação devem conter a coluna 'contem_dados_pessoais'")
        sys.exit(1)

    # Validar qualidade
    logger.info("\n3. Validando qualidade dos dados...")
    df = validar_qualidade_dados(df)

    # Preprocessar
    logger.info("\n4. Preprocessando dados...")
    df = preprocessar_dados(df, coluna_texto='texto_pedido', criar_features=True)

    # Fazer predições
    logger.info("\n5. Fazendo predições...")
    y_true = df['contem_dados_pessoais'].values
    y_pred = modelo.prever_batch(df['texto_limpo'])

    # Avaliar
    logger.info("\n6. Calculando métricas...")
    metricas = avaliar_modelo(y_true, y_pred, mostrar_detalhes=True)

    # Análise de erros
    if args.analisar_erros:
        logger.info("\n7. Analisando erros...")
        erros = analisar_erros(
            df,
            y_true,
            y_pred,
            coluna_texto='texto_pedido',
            top_n=10
        )

        # Salvar erros
        if args.salvar_erros:
            caminho_fp = Path(args.output_dir) / "falsos_positivos.csv"
            caminho_fn = Path(args.output_dir) / "falsos_negativos.csv"

            erros['falsos_positivos'].to_csv(caminho_fp, index=False)
            erros['falsos_negativos'].to_csv(caminho_fn, index=False)

            logger.info(f"\nErros salvos em:")
            logger.info(f"  Falsos Positivos: {caminho_fp}")
            logger.info(f"  Falsos Negativos: {caminho_fn}")

    # Plotar matriz de confusão
    if args.plotar:
        logger.info("\n8. Plotando matriz de confusão...")
        caminho_plot = Path(args.output_dir) / "matriz_confusao_avaliacao.png"
        plotar_matriz_confusao(y_true, y_pred, salvar_caminho=str(caminho_plot))

    # Salvar relatório
    logger.info("\n9. Salvando relatório...")
    caminho_relatorio = Path(args.output_dir) / "relatorio_avaliacao_final.json"
    salvar_relatorio(
        metricas,
        str(caminho_relatorio),
        adicionar_info={
            'modelo': args.model,
            'dados_teste': args.data
        }
    )

    logger.info("\n" + "="*60)
    logger.info("AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
    logger.info("="*60)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Sistema de Detecção de Dados Pessoais - Hackathon Participa DF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Treinar modelo
  python main.py --mode train --data dados/treino.csv --output modelos/modelo.pkl

  # Fazer predições
  python main.py --mode predict --model modelos/modelo.pkl --input dados/teste.csv --output resultados/predicoes.csv

  # Avaliar modelo
  python main.py --mode evaluate --model modelos/modelo.pkl --data dados/teste_rotulado.csv
        """
    )

    # Argumentos principais
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'predict', 'evaluate'],
        help='Modo de operação: train (treinar), predict (prever), evaluate (avaliar)'
    )

    # Argumentos de dados
    parser.add_argument(
        '--data',
        type=str,
        help='Caminho para arquivo CSV de dados (treino ou avaliação)'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Caminho para arquivo CSV de entrada (modo predict)'
    )

    # Argumentos de modelo
    parser.add_argument(
        '--model',
        type=str,
        default='modelos/modelo_treinado.pkl',
        help='Caminho para o modelo (carregar ou salvar)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='modelos/modelo_treinado.pkl',
        help='Caminho para arquivo de saída'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='resultados',
        help='Diretório para salvar resultados (modo evaluate)'
    )

    # Argumentos de configuração
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold de confiança para classificação (padrão: 0.5)'
    )

    parser.add_argument(
        '--validacao',
        action='store_true',
        help='Usar split treino/validação durante treinamento'
    )

    # Argumentos de análise
    parser.add_argument(
        '--analisar-erros',
        action='store_true',
        help='Analisar falsos positivos e falsos negativos'
    )

    parser.add_argument(
        '--salvar-erros',
        action='store_true',
        help='Salvar exemplos de erros em CSV'
    )

    parser.add_argument(
        '--salvar-relatorio',
        action='store_true',
        help='Salvar relatório de avaliação'
    )

    parser.add_argument(
        '--plotar',
        action='store_true',
        help='Plotar matriz de confusão'
    )

    args = parser.parse_args()

    # Validar argumentos
    if args.mode == 'train':
        if not args.data:
            parser.error("--data é obrigatório no modo train")
    elif args.mode == 'predict':
        if not args.input or not args.model:
            parser.error("--input e --model são obrigatórios no modo predict")
    elif args.mode == 'evaluate':
        if not args.data or not args.model:
            parser.error("--data e --model são obrigatórios no modo evaluate")

    # Executar modo apropriado
    try:
        if args.mode == 'train':
            treinar_modelo(args)
        elif args.mode == 'predict':
            prever(args)
        elif args.mode == 'evaluate':
            avaliar(args)
    except Exception as e:
        logger.error(f"\nERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
