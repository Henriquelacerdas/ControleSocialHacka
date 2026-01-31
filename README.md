# Solução para Identificação de Dados Pessoais em Pedidos de Acesso à Informação
## Hackathon Participa DF - Categoria Acesso à Informação

### Descrição do Projeto

Este projeto implementa um sistema automatizado para identificar pedidos de acesso à informação que contenham dados pessoais (nome, CPF, RG, telefone, e-mail) e que, portanto, deveriam ser classificados como "não públicos" em vez de "públicos".

A solução utiliza uma abordagem híbrida em múltiplas camadas:
1. **Detecção por padrões (Regex)**: Identificação de CPF, RG, telefones e e-mails através de expressões regulares robustas
2. **Named Entity Recognition (NER)**: Reconhecimento de nomes próprios de pessoas
3. **Machine Learning**: Classificação complementar usando TF-IDF e algoritmos de ML tradicionais

Esta estratégia garante alta precisão e recall, priorizando a identificação de todos os pedidos que contenham dados pessoais (minimizando falsos negativos) enquanto mantém baixo o número de falsos positivos.

---

## Pré-requisitos

- **Python 3.9+** (recomendado: Python 3.9, 3.10 ou 3.11)
- **pip** (gerenciador de pacotes Python)
- **git** (para clonar o repositório)
- Conexão com a internet (para download de modelos NER do spaCy na primeira execução)

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/hackathon-participa-df.git
cd hackathon-participa-df
```

### 2. Crie um ambiente virtual

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Baixe o modelo de NER em português do spaCy

```bash
python -m spacy download pt_core_news_lg
```

---

## Estrutura do Projeto

```
hackathon-participa-df/
├── README.md                    # Documentação principal (este arquivo)
├── requirements.txt             # Lista de dependências Python
├── .gitignore                   # Arquivos ignorados pelo Git
│
├── dados/                       # Pasta para dados (não commitados)
│   ├── treino/                  # Dados de treinamento
│   └── teste/                   # Dados de teste
│
├── src/                         # Código-fonte principal
│   ├── preprocessamento.py      # Limpeza e preparação dos dados
│   ├── modelo.py                # Implementação do modelo híbrido
│   ├── avaliacao.py             # Cálculo de métricas (F1-Score, etc.)
│   └── main.py                  # Script principal de execução
│
├── notebooks/                   # Jupyter notebooks
│   └── exploracao.ipynb         # Análise exploratória dos dados
│
├── modelos/                     # Modelos treinados salvos (.pkl, .joblib)
└── resultados/                  # Outputs, previsões, métricas
```

### Descrição dos Arquivos Principais

- **[src/preprocessamento.py](src/preprocessamento.py)**: Funções para carregar, limpar e normalizar os textos dos pedidos
- **[src/modelo.py](src/modelo.py)**: Implementação da detecção híbrida (regex, NER, ML)
- **[src/avaliacao.py](src/avaliacao.py)**: Cálculo de métricas de desempenho (Precisão, Recall, F1-Score)
- **[src/main.py](src/main.py)**: Interface de linha de comando para treinar e fazer predições

---

## Como Executar

### Treinamento do Modelo

Para treinar o modelo com seus dados de treinamento:

```bash
python src/main.py --mode train --data dados/treino/pedidos_treino.csv --output modelos/modelo_treinado.pkl
```

**Parâmetros:**
- `--mode train`: Modo de treinamento
- `--data`: Caminho para o arquivo CSV de treinamento
- `--output`: Caminho onde o modelo treinado será salvo (opcional, padrão: `modelos/modelo_treinado.pkl`)

### Inferência/Previsão

Para fazer previsões em novos dados:

```bash
python src/main.py --mode predict --model modelos/modelo_treinado.pkl --input dados/teste/pedidos_teste.csv --output resultados/predicoes.csv
```

**Parâmetros:**
- `--mode predict`: Modo de predição
- `--model`: Caminho para o modelo treinado
- `--input`: Caminho para o arquivo CSV com os pedidos a serem analisados
- `--output`: Caminho onde as previsões serão salvas

### Avaliação de Performance

Para avaliar o modelo em um conjunto de teste com rótulos conhecidos:

```bash
python src/main.py --mode evaluate --model modelos/modelo_treinado.pkl --data dados/teste/pedidos_teste_rotulados.csv
```

**Parâmetros:**
- `--mode evaluate`: Modo de avaliação
- `--model`: Caminho para o modelo treinado
- `--data`: Caminho para o arquivo CSV de teste com rótulos verdadeiros

---

## Formato dos Dados

### Entrada Esperada

Arquivo CSV com as seguintes colunas obrigatórias:

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `id_pedido` | int/string | Identificador único do pedido | 12345 |
| `texto_pedido` | string | Texto completo do pedido | "Solicito informações sobre..." |
| `contem_dados_pessoais` | bool | Rótulo verdadeiro (apenas para treino/avaliação) | True/False |

**Exemplo de CSV de entrada:**
```csv
id_pedido,texto_pedido,contem_dados_pessoais
1,"Gostaria de saber o andamento do processo 123.456.789-00",True
2,"Solicito informações sobre o orçamento de 2025",False
```

### Saída Gerada

Arquivo CSV com as previsões do modelo:

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `id_pedido` | int/string | Identificador único do pedido | 12345 |
| `contem_dados_pessoais` | bool | Predição do modelo | True/False |
| `confianca` | float | Score de confiança (0-1) | 0.95 |

**Exemplo de CSV de saída:**
```csv
id_pedido,contem_dados_pessoais,confianca
1,True,0.98
2,False,0.87
```

---

## Metodologia e Abordagem Técnica

### Estratégia de Detecção em Múltiplas Camadas

#### Camada 1: Detecção por Padrões (Regex)
Expressões regulares robustas para identificar:
- **CPF**: Formatos XXX.XXX.XXX-XX, XXX XXX XXX XX, XXXXXXXXXXX
- **RG**: Diversos formatos estaduais (SP, RJ, MG, DF, etc.)
- **Telefone**: Celular e fixo, com ou sem DDD, formatos variados
- **E-mail**: Todos os formatos válidos de e-mail

#### Camada 2: Named Entity Recognition (NER)
Utiliza o modelo `pt_core_news_lg` do spaCy para:
- Identificar nomes próprios de pessoas (entidade PERSON/PER)
- Filtrar falsos positivos (nomes de lugares, organizações)

#### Camada 3: Machine Learning Tradicional
- **Feature Engineering**: TF-IDF, contagem de números, padrões específicos
- **Algoritmo**: Ensemble de Random Forest e Logistic Regression
- **Treinamento**: Balanceamento de classes e validação cruzada

#### Estratégia de Ensemble
As camadas são combinadas usando um sistema de votação ponderada:
- Se regex detecta padrão de dado pessoal → peso alto
- Se NER identifica nome próprio → peso médio-alto
- Se ML classifica como contendo dados → peso médio
- Threshold ajustado para priorizar recall (minimizar falsos negativos)

### Por que esta abordagem?

1. **Alta Robustez**: Regex captura padrões óbvios, NER captura nomes que regex não pega
2. **Baixo Falso Negativo**: Múltiplas camadas reduzem chance de perder dados pessoais
3. **Interpretabilidade**: Conseguimos explicar porque um pedido foi classificado
4. **Performance**: Não requer GPU, roda em qualquer máquina

---

## Tecnologias Utilizadas

### Principais Bibliotecas

- **Python 3.9+**: Linguagem de programação principal
- **pandas 2.1.4**: Manipulação de dados tabulares
- **scikit-learn 1.3.2**: Algoritmos de machine learning
- **spacy 3.7.2**: Processamento de linguagem natural e NER
- **nltk 3.8.1**: Tokenização e processamento de texto
- **regex 2023.10.3**: Expressões regulares avançadas
- **python-stdnum 1.19**: Validação de CPF e documentos brasileiros

### Ferramentas de Desenvolvimento

- **jupyter 1.0.0**: Análise exploratória interativa
- **matplotlib 3.8.2** / **seaborn 0.13.0**: Visualização de dados
- **tqdm 4.66.1**: Barras de progresso
- **joblib 1.3.2**: Serialização de modelos

---

## Métricas de Performance

As métricas são calculadas usando a fórmula oficial do hackathon:

**Precisão = VP / (VP + FP)**
- VP = Verdadeiros Positivos (pedidos corretamente identificados como contendo dados pessoais)
- FP = Falsos Positivos (pedidos incorretamente identificados)

**Recall/Sensibilidade = VP / (VP + FN)**
- FN = Falsos Negativos (pedidos com dados pessoais não identificados)

**F1-Score = 2 × (Precisão × Recall) / (Precisão + Recall)**

### Resultados em Validação Cruzada

*(Os resultados abaixo serão atualizados após treinamento com dados reais)*

```
Precisão:  XX.X%
Recall:    XX.X%
F1-Score:  XX.X%

Falsos Positivos:  XX
Falsos Negativos:  XX
```

---

## Exemplos de Uso

### Exemplo 1: Pipeline Completo

```bash
# 1. Treinar modelo
python src/main.py --mode train --data dados/treino/pedidos.csv

# 2. Fazer previsões
python src/main.py --mode predict --input dados/teste/novos_pedidos.csv --output resultados/predicoes.csv

# 3. Avaliar performance
python src/main.py --mode evaluate --data dados/teste/pedidos_rotulados.csv
```

### Exemplo 2: Usando Python Diretamente

```python
from src.modelo import ModeloHibridoDeteccao
from src.preprocessamento import carregar_dados, preprocessar_texto
import pandas as pd

# Carregar e treinar modelo
dados_treino = carregar_dados('dados/treino/pedidos.csv')
modelo = ModeloHibridoDeteccao()
modelo.treinar(dados_treino)

# Fazer predições
novos_pedidos = pd.read_csv('dados/teste/novos_pedidos.csv')
predicoes = modelo.prever(novos_pedidos['texto_pedido'])

# Salvar resultados
resultado = pd.DataFrame({
    'id_pedido': novos_pedidos['id_pedido'],
    'contem_dados_pessoais': predicoes
})
resultado.to_csv('resultados/predicoes.csv', index=False)
```

---

## Tratamento de Variações

O sistema é robusto para lidar com diversas variações de formatação:

### CPF
- Com pontuação: `123.456.789-00`
- Sem pontuação: `12345678900`
- Com espaços: `123 456 789 00`

### Telefone
- Com DDD e hífen: `(61) 98765-4321`
- Sem formatação: `61987654321`
- Com código país: `+55 61 9 8765-4321`
- Fixo: `(61) 3456-7890`

### RG
- Diversos formatos estaduais: `12.345.678-9` (SP), `1.234.567` (RJ)
- Com e sem pontuação

### E-mail
- Todos os formatos válidos: `usuario@exemplo.com.br`, `nome.sobrenome@dominio.org`

### Nomes
- Com títulos: `Dr. João Silva`, `Sr. Pedro Santos`
- Com abreviações: `José A. Oliveira`
- Nomes compostos: `Maria da Silva Santos`

---

## Solução de Problemas

### Erro ao importar spacy

```bash
# Certifique-se de baixar o modelo de NER em português
python -m spacy download pt_core_news_lg
```

### Erro de memória ao processar muitos dados

No arquivo [src/main.py](src/main.py), ajuste o parâmetro `batch_size`:

```python
modelo.prever_batch(dados, batch_size=100)  # Reduzir se necessário
```

### Performance lenta

Para grandes volumes de dados, considere:
- Usar processamento em paralelo (já implementado)
- Aumentar `batch_size` se houver memória disponível
- Desativar camadas opcionais no modelo

---

## Desenvolvimento e Testes

### Executar Análise Exploratória

```bash
jupyter notebook notebooks/exploracao.ipynb
```

### Executar Testes Unitários

```bash
# (Testes serão adicionados conforme necessário)
python -m pytest tests/
```

---

## Autores

**Henrique Lacerda Silveira**
- GitHub: [@seu-usuario](https://github.com/seu-usuario)

Projeto desenvolvido para o **1º Hackathon em Controle Social: Desafio Participa DF**
Categoria: Acesso à Informação
Organizador: Controladoria-Geral do Distrito Federal (CGDF)

---

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

**Nota**: Conforme regras do hackathon, a propriedade intelectual desta solução será transferida para a Controladoria-Geral do Distrito Federal (CGDF) caso seja premiada.

---

## Agradecimentos

- Controladoria-Geral do Distrito Federal (CGDF) pela organização do hackathon
- Comunidade Python e spaCy pelos excelentes recursos de NLP
- Todos os participantes que contribuem para o controle social e transparência pública

---

## Contato e Suporte

Para dúvidas, sugestões ou reportar problemas:
- Abra uma issue no GitHub: [github.com/seu-usuario/hackathon-participa-df/issues](https://github.com/seu-usuario/hackathon-participa-df/issues)
- Entre em contato: seu-email@exemplo.com

---

**Última atualização**: 31/01/2026
**Versão**: 1.0.0
