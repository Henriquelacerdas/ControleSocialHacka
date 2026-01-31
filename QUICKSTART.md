# Quick Start - Início Rápido

## Instalação Rápida (5 minutos)

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/hackathon-participa-df.git
cd hackathon-participa-df

# 2. Crie ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# 3. Instale dependências
pip install --upgrade pip
pip install -r requirements.txt

# 4. Baixe modelo de NER em português
python -m spacy download pt_core_news_lg
```

## Teste Rápido

```bash
# Execute o script de exemplo
python exemplo_uso.py
```

## Uso com Dados Reais

### 1. Obter Dados da CGDF

Baixe os dados do hackathon de: https://www.cg.df.gov.br/

Coloque os arquivos em:
- `dados/treino/pedidos_treino.csv`
- `dados/teste/pedidos_teste.csv`

### 2. Treinar Modelo

```bash
python src/main.py --mode train \
    --data dados/treino/pedidos_treino.csv \
    --output modelos/modelo_treinado.pkl \
    --validacao \
    --analisar-erros \
    --plotar
```

### 3. Fazer Predições

```bash
python src/main.py --mode predict \
    --model modelos/modelo_treinado.pkl \
    --input dados/teste/pedidos_teste.csv \
    --output resultados/predicoes.csv
```

### 4. Avaliar Performance

```bash
python src/main.py --mode evaluate \
    --model modelos/modelo_treinado.pkl \
    --data dados/teste/pedidos_teste_rotulados.csv \
    --analisar-erros \
    --plotar
```

## Análise Exploratória

```bash
# Abrir notebook Jupyter
jupyter notebook notebooks/exploracao.ipynb
```

## Estrutura Mínima de Dados

### Entrada (CSV)
```csv
id_pedido,texto_pedido,contem_dados_pessoais
1,"Solicito informações sobre...",False
2,"Meu CPF é 123.456.789-00...",True
```

### Saída (CSV)
```csv
id_pedido,contem_dados_pessoais
1,False
2,True
```

## Arquivos Importantes

- **[README.md](README.md)**: Documentação completa
- **[SUBMISSAO.md](SUBMISSAO.md)**: Checklist e guia de submissão
- **[exemplo_uso.py](exemplo_uso.py)**: Exemplos práticos de uso
- **[notebooks/exploracao.ipynb](notebooks/exploracao.ipynb)**: Análise exploratória

## Ajuda

```bash
# Ver opções disponíveis
python src/main.py --help
```

## Problemas Comuns

### Erro ao importar spacy
```bash
python -m spacy download pt_core_news_lg
```

### Erro de memória
Reduza o batch_size no código ou use menos dados

### ModuleNotFoundError
```bash
pip install -r requirements.txt
```

## Próximos Passos

1. Leia o [README.md](README.md) completo
2. Execute os exemplos em [exemplo_uso.py](exemplo_uso.py)
3. Explore os dados no notebook [exploracao.ipynb](notebooks/exploracao.ipynb)
4. Treine e otimize seu modelo
5. Siga o checklist em [SUBMISSAO.md](SUBMISSAO.md) antes de submeter

## Suporte

- Issues: [GitHub Issues](https://github.com/seu-usuario/hackathon-participa-df/issues)
- Documentação: [README.md](README.md)
- CGDF: https://www.cg.df.gov.br/

---

**Boa sorte no hackathon! 🚀**
