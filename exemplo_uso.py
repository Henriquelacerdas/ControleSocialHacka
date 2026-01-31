#!/usr/bin/env python3
"""
Script de Exemplo de Uso
Demonstra como usar o sistema de detecção de dados pessoais
"""

import pandas as pd
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from modelo import ModeloHibridoDeteccao, DetectorPadroesRegex
from preprocessamento import preprocessar_dados


def exemplo_deteccao_rapida():
    """Exemplo 1: Detecção rápida usando apenas regex."""
    print("="*60)
    print("EXEMPLO 1: Detecção Rápida com Regex")
    print("="*60)

    # Criar detector
    detector = DetectorPadroesRegex()

    # Textos de teste
    textos_teste = [
        "Solicito informações sobre o orçamento de 2025",
        "Meu CPF é 123.456.789-00 e preciso de dados do processo",
        "Entre em contato: joao.silva@email.com",
        "Telefone para contato: (61) 98765-4321",
        "Pedro Santos solicita informações",
        "Qual o status da licitação 2024/001?",
    ]

    print("\nAnalisando textos:\n")
    for i, texto in enumerate(textos_teste, 1):
        deteccoes = detector.detectar_todos(texto)
        contem = detector.contem_dados_pessoais(texto)

        print(f"{i}. '{texto}'")
        print(f"   Contém dados pessoais: {'SIM' if contem else 'NÃO'}")
        if contem:
            tipos = [k for k, v in deteccoes.items() if v]
            print(f"   Tipos detectados: {', '.join(tipos)}")
        print()


def exemplo_modelo_completo():
    """Exemplo 2: Usando o modelo híbrido completo."""
    print("="*60)
    print("EXEMPLO 2: Modelo Híbrido Completo")
    print("="*60)

    # Criar dados de exemplo
    print("\n1. Criando dados de exemplo...")
    df_treino = pd.DataFrame({
        'id_pedido': range(1, 11),
        'texto_pedido': [
            "Solicito informações sobre o orçamento público de 2025",
            "Meu CPF é 123.456.789-00, preciso de informações",
            "Entre em contato: joao@email.com",
            "Qual o status das obras no DF?",
            "Telefone: (61) 98765-4321",
            "Relatório de despesas públicas",
            "Maria Silva solicita dados do processo",
            "Informações sobre licitações abertas",
            "RG 1.234.567 para consulta",
            "Transparência nos gastos públicos"
        ],
        'contem_dados_pessoais': [False, True, True, False, True, False, True, False, True, False]
    })

    print(f"   Dados criados: {len(df_treino)} amostras")

    # Preprocessar
    print("\n2. Preprocessando dados...")
    df_treino = preprocessar_dados(df_treino)

    # Criar e treinar modelo
    print("\n3. Treinando modelo híbrido...")
    modelo = ModeloHibridoDeteccao(
        usar_regex=True,
        usar_ner=True,
        usar_ml=True,
        threshold_confianca=0.5
    )

    modelo.treinar(
        df_treino,
        coluna_texto='texto_limpo',
        coluna_label='contem_dados_pessoais'
    )

    print("   Modelo treinado com sucesso!")

    # Fazer predições
    print("\n4. Testando predições...")
    textos_teste = [
        "Solicito relatório de gastos",
        "CPF 987.654.321-00 para consulta",
        "Contato: pedro@exemplo.com.br"
    ]

    for texto in textos_teste:
        # Preprocessar
        df_teste = pd.DataFrame({'texto_pedido': [texto]})
        df_teste = preprocessar_dados(df_teste)

        # Prever
        predicao, detalhes = modelo.prever(
            df_teste['texto_limpo'].iloc[0],
            retornar_detalhes=True
        )

        print(f"\n   Texto: '{texto}'")
        print(f"   Predição: {'CONTÉM dados pessoais' if predicao else 'NÃO contém dados pessoais'}")
        print(f"   Confiança: {detalhes['score_final']:.2f}")

    # Salvar modelo
    print("\n5. Salvando modelo...")
    modelo.salvar('modelos/modelo_exemplo.pkl')
    print("   Modelo salvo em: modelos/modelo_exemplo.pkl")


def exemplo_uso_modelo_salvo():
    """Exemplo 3: Carregar e usar modelo salvo."""
    print("\n" + "="*60)
    print("EXEMPLO 3: Usando Modelo Salvo")
    print("="*60)

    caminho_modelo = 'modelos/modelo_exemplo.pkl'

    if not Path(caminho_modelo).exists():
        print(f"\nModelo não encontrado em: {caminho_modelo}")
        print("Execute primeiro o Exemplo 2 para criar o modelo.")
        return

    # Carregar modelo
    print("\n1. Carregando modelo...")
    modelo = ModeloHibridoDeteccao.carregar(caminho_modelo)
    print("   Modelo carregado com sucesso!")

    # Fazer predições
    print("\n2. Fazendo predições...")
    novos_textos = [
        "Informações sobre transparência pública",
        "Meu nome é João Silva e CPF 111.222.333-44",
        "Email: contato@exemplo.com para retorno"
    ]

    # Preprocessar
    df_novo = pd.DataFrame({'texto_pedido': novos_textos})
    df_novo = preprocessar_dados(df_novo)

    # Prever em batch
    predicoes = modelo.prever_batch(df_novo['texto_limpo'])

    print("\nResultados:")
    for texto, pred in zip(novos_textos, predicoes):
        print(f"   '{texto[:50]}...' -> {'CONTÉM' if pred else 'NÃO CONTÉM'}")


def main():
    """Função principal."""
    print("\n" + "#"*60)
    print("# Sistema de Detecção de Dados Pessoais")
    print("# Hackathon Participa DF - Exemplos de Uso")
    print("#"*60 + "\n")

    try:
        # Exemplo 1: Detecção rápida
        exemplo_deteccao_rapida()

        # Exemplo 2: Modelo completo
        input("\nPressione ENTER para continuar para o Exemplo 2...")
        exemplo_modelo_completo()

        # Exemplo 3: Modelo salvo
        input("\nPressione ENTER para continuar para o Exemplo 3...")
        exemplo_uso_modelo_salvo()

        print("\n" + "="*60)
        print("TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
        print("="*60)
        print("\nPróximos passos:")
        print("1. Obtenha os dados reais da CGDF")
        print("2. Execute: python src/main.py --mode train --data dados/treino.csv")
        print("3. Faça predições: python src/main.py --mode predict --input dados/teste.csv")
        print("\nConsulte o README.md para mais informações.")

    except KeyboardInterrupt:
        print("\n\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n\nERRO: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
