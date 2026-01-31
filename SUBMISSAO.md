# Guia de Submissão - Hackathon Participa DF

## Checklist Antes de Submeter

Marque cada item conforme completar:

### Documentação (Pontuação P2 - 10 pontos)

#### Instruções de Instalação e Dependências (4 pontos)
- [ ] Lista de pré-requisitos clara no README.md (Python 3.9+, pip, git)
- [ ] Arquivo requirements.txt presente e completo
- [ ] Comandos exatos para criar e configurar ambiente virtual
- [ ] Instruções para baixar modelo spaCy (python -m spacy download pt_core_news_lg)

#### Instruções de Execução (3 pontos)
- [ ] Comandos exatos para treinar o modelo com exemplos
- [ ] Comandos exatos para fazer predições com exemplos
- [ ] Formato de entrada descrito (CSV com id_pedido e texto_pedido)
- [ ] Formato de saída descrito (CSV com id_pedido e contem_dados_pessoais)

#### Clareza e Organização (3 pontos)
- [ ] README.md descreve objetivo e função de cada arquivo
- [ ] Código-fonte tem comentários em trechos complexos
- [ ] Estrutura de arquivos lógica e bem organizada

### Código e Funcionalidade

- [ ] Repositório GitHub criado e público
- [ ] Código está commitado e funcionando
- [ ] .gitignore configurado (não commitar dados/modelos grandes)
- [ ] Arquivos .gitkeep nas pastas vazias para manter estrutura
- [ ] Todos os imports funcionam corretamente
- [ ] Scripts executam sem erros em ambiente limpo

### Testes Finais

- [ ] Testado comando de instalação em ambiente limpo:
  ```bash
  python -m venv venv
  source venv/bin/activate  # ou venv\Scripts\activate no Windows
  pip install -r requirements.txt
  python -m spacy download pt_core_news_lg
  ```

- [ ] Testado comando de treinamento:
  ```bash
  python src/main.py --mode train --data dados/treino.csv --output modelos/modelo.pkl --validacao
  ```

- [ ] Testado comando de predição:
  ```bash
  python src/main.py --mode predict --model modelos/modelo.pkl --input dados/teste.csv --output resultados/predicoes.csv
  ```

- [ ] Testado comando de avaliação:
  ```bash
  python src/main.py --mode evaluate --model modelos/modelo.pkl --data dados/teste_rotulado.csv
  ```

### Repositório GitHub

- [ ] URL do repositório: ________________________________
- [ ] Repositório está PÚBLICO
- [ ] README.md visível na página inicial
- [ ] Estrutura de pastas visível e organizada
- [ ] Última atualização dentro do prazo (até 30/01/2026)

---

## Passos para Submissão

### 1. Preparar Repositório GitHub

```bash
# Inicializar git (se ainda não inicializado)
git init

# Adicionar todos os arquivos
git add .

# Fazer commit inicial
git commit -m "Solução inicial para Hackathon Participa DF - Detecção de Dados Pessoais"

# Criar repositório no GitHub (via interface web)
# Depois conectar e fazer push:
git remote add origin https://github.com/SEU-USUARIO/hackathon-participa-df.git
git branch -M main
git push -u origin main
```

### 2. Verificar Tudo Está Funcionando

```bash
# Clone em outro diretório para testar
cd /tmp
git clone https://github.com/SEU-USUARIO/hackathon-participa-df.git
cd hackathon-participa-df

# Seguir instruções do README.md passo a passo
# Verificar se tudo funciona
```

### 3. Preencher Formulário de Submissão

Acesse o formulário oficial da CGDF e preencha:

- **Nome do Projeto**: Sistema de Detecção de Dados Pessoais em Pedidos de Acesso à Informação
- **Categoria**: Acesso à Informação
- **URL do Repositório GitHub**: https://github.com/SEU-USUARIO/hackathon-participa-df
- **Descrição Breve**: (Copiar do README.md)
- **Tecnologias Utilizadas**: Python, scikit-learn, spaCy, pandas, regex
- **Dados da Equipe**: (Seu nome e informações)

### 4. Confirmação

- [ ] Formulário submetido com sucesso
- [ ] E-mail de confirmação recebido
- [ ] Data e hora de submissão: ____/____/2026 às ____:____

---

## Informações Importantes

### Prazo de Submissão
- **Início**: 12/01/2026
- **Término**: 30/01/2026 às 23:59
- **IMPORTANTE**: Commits após a submissão do formulário NÃO serão considerados

### Avaliação
- **Período**: 02/02 a 20/02/2026
- **Resultado**: 23/02/2026

### Critérios de Avaliação
1. **P1 - Desempenho (F1-Score)**: Peso maior
   - Fórmula: 2 × (Precisão × Recall) / (Precisão + Recall)
   - Desempate: Menor FN > Menor FP > Maior P1

2. **P2 - Documentação**: Máximo 10 pontos
   - Instalação: 4 pontos
   - Execução: 3 pontos
   - Clareza: 3 pontos

### Premiação
- **1º lugar**: R$ 8.000,00
- **2º lugar**: R$ 5.000,00
- **3º lugar**: R$ 2.000,00

---

## Contatos e Suporte

- **Site da CGDF**: https://www.cg.df.gov.br/
- **Dados do Hackathon**: [Link será fornecido pela CGDF]
- **Dúvidas**: [E-mail de contato da CGDF]

---

## Após a Submissão

### O que fazer enquanto espera o resultado:

1. **Não altere o repositório** (commits não serão considerados)
2. Prepare apresentação (se houver fase de apresentação)
3. Documente aprendizados e melhorias futuras
4. Continue estudando sobre controle social e transparência pública

### Se for selecionado:

1. Esteja disponível para contato da CGDF
2. Prepare documentação adicional se solicitada
3. Esteja pronto para explicar sua solução
4. Entenda que a propriedade intelectual será transferida para a CGDF

---

## Boa Sorte! 🚀

Você desenvolveu uma solução completa, bem documentada e competitiva.
Agora é hora de submeter e aguardar o resultado!

**Lembre-se**: O mais importante é a experiência de aprendizado e a contribuição
para o controle social e transparência pública no Distrito Federal.

---

**Data deste documento**: 31/01/2026
**Versão**: 1.0
