"""
Módulo do Modelo de Detecção de Dados Pessoais
Hackathon Participa DF - Categoria Acesso à Informação

Este módulo implementa uma abordagem híbrida em múltiplas camadas:
1. Detecção por padrões (Regex) - CPF, RG, telefone, e-mail
2. Named Entity Recognition (NER) - Nomes de pessoas
3. Machine Learning - Classificação complementar
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import joblib
from pathlib import Path

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Validação de documentos brasileiros
try:
    from stdnum.br import cpf as cpf_validator
except ImportError:
    cpf_validator = None

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DetectorPadroesRegex:
    """
    Detector de dados pessoais usando expressões regulares.
    """

    def __init__(self):
        """Inicializa os padrões regex para cada tipo de dado pessoal."""

        # Padrão de CPF - diversos formatos
        self.pattern_cpf = [
            r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',  # 123.456.789-00 ou 12345678900
            r'\b\d{3}\s\d{3}\s\d{3}\s\d{2}\b',    # 123 456 789 00
        ]

        # Padrão de RG - formatos variados por estado
        self.pattern_rg = [
            r'\b\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b',  # 12.345.678-9 ou similar
            r'\bRG:?\s*\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b',  # RG: 12.345.678-9
            r'\b\d{7,9}\b',  # Sequência de 7-9 dígitos (pode ser RG)
        ]

        # Padrão de telefone - celular e fixo
        self.pattern_telefone = [
            r'\b\(?0?\d{2}\)?\s?9?\d{4}-?\d{4}\b',  # (61) 98765-4321 ou variações
            r'\b\+?55\s?\(?0?\d{2}\)?\s?9?\d{4}-?\d{4}\b',  # +55 61 98765-4321
            r'\b\d{10,11}\b',  # 61987654321
        ]

        # Padrão de e-mail
        self.pattern_email = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ]

        # Compilar todos os padrões
        self.compiled_patterns = {
            'cpf': [re.compile(p, re.IGNORECASE) for p in self.pattern_cpf],
            'rg': [re.compile(p, re.IGNORECASE) for p in self.pattern_rg],
            'telefone': [re.compile(p, re.IGNORECASE) for p in self.pattern_telefone],
            'email': [re.compile(p, re.IGNORECASE) for p in self.pattern_email],
        }

    def detectar_cpf(self, texto: str) -> bool:
        """
        Detecta CPF no texto.

        Args:
            texto: Texto a ser analisado

        Returns:
            True se CPF detectado, False caso contrário
        """
        if pd.isna(texto):
            return False

        texto = str(texto)

        # Buscar padrões de CPF
        for pattern in self.compiled_patterns['cpf']:
            matches = pattern.findall(texto)
            if matches:
                # Validar CPF se possível
                for match in matches:
                    cpf_numeros = re.sub(r'\D', '', match)
                    if len(cpf_numeros) == 11:
                        # Verificação básica: não pode ser todos números iguais
                        if len(set(cpf_numeros)) > 1:
                            # Se temos validador, usar
                            if cpf_validator:
                                try:
                                    if cpf_validator.is_valid(cpf_numeros):
                                        return True
                                except:
                                    # Se falhar validação, considerar válido por padrão
                                    return True
                            else:
                                return True

        return False

    def detectar_rg(self, texto: str) -> bool:
        """Detecta RG no texto."""
        if pd.isna(texto):
            return False

        texto = str(texto)

        # Buscar menção explícita de RG
        if re.search(r'\bRG\b', texto, re.IGNORECASE):
            # Se tem a palavra RG, buscar padrões próximos
            for pattern in self.compiled_patterns['rg']:
                if pattern.search(texto):
                    return True

        return False

    def detectar_telefone(self, texto: str) -> bool:
        """Detecta telefone no texto."""
        if pd.isna(texto):
            return False

        texto = str(texto)

        # Buscar padrões de telefone
        for pattern in self.compiled_patterns['telefone']:
            matches = pattern.findall(texto)
            if matches:
                # Filtrar falsos positivos (ex: sequências que não são telefones)
                for match in matches:
                    numeros = re.sub(r'\D', '', match)
                    # Telefone brasileiro tem 10 ou 11 dígitos
                    if len(numeros) in [10, 11]:
                        # Verificar se começa com DDD válido (1-9)
                        if numeros[0] in '123456789':
                            return True

        return False

    def detectar_email(self, texto: str) -> bool:
        """Detecta e-mail no texto."""
        if pd.isna(texto):
            return False

        texto = str(texto)

        for pattern in self.compiled_patterns['email']:
            if pattern.search(texto):
                return True

        return False

    def detectar_todos(self, texto: str) -> Dict[str, bool]:
        """
        Detecta todos os tipos de dados pessoais no texto.

        Args:
            texto: Texto a ser analisado

        Returns:
            Dicionário com resultado de cada tipo de detecção
        """
        return {
            'cpf': self.detectar_cpf(texto),
            'rg': self.detectar_rg(texto),
            'telefone': self.detectar_telefone(texto),
            'email': self.detectar_email(texto),
        }

    def contem_dados_pessoais(self, texto: str) -> bool:
        """
        Verifica se o texto contém qualquer tipo de dado pessoal.

        Args:
            texto: Texto a ser analisado

        Returns:
            True se contém algum dado pessoal, False caso contrário
        """
        deteccoes = self.detectar_todos(texto)
        return any(deteccoes.values())


class DetectorNER:
    """
    Detector de nomes de pessoas usando Named Entity Recognition (NER).
    """

    def __init__(self, modelo_spacy: str = 'pt_core_news_lg'):
        """
        Inicializa o detector NER.

        Args:
            modelo_spacy: Nome do modelo spaCy a ser usado
        """
        self.modelo_nome = modelo_spacy
        self.nlp = None

    def carregar_modelo(self):
        """Carrega o modelo spaCy (lazy loading)."""
        if self.nlp is None:
            try:
                import spacy
                logger.info(f"Carregando modelo spaCy: {self.modelo_nome}")
                self.nlp = spacy.load(self.modelo_nome)
                logger.info("Modelo spaCy carregado com sucesso!")
            except OSError:
                logger.error(f"Modelo spaCy '{self.modelo_nome}' não encontrado.")
                logger.error("Execute: python -m spacy download pt_core_news_lg")
                raise
            except ImportError:
                logger.error("spaCy não está instalado. Execute: pip install spacy")
                raise

    def detectar_nomes(self, texto: str) -> List[str]:
        """
        Detecta nomes de pessoas no texto.

        Args:
            texto: Texto a ser analisado

        Returns:
            Lista de nomes detectados
        """
        if pd.isna(texto):
            return []

        # Carregar modelo se necessário
        if self.nlp is None:
            self.carregar_modelo()

        texto = str(texto)

        # Processar texto com spaCy
        doc = self.nlp(texto)

        # Extrair entidades do tipo PERSON (PER em alguns modelos)
        nomes = []
        for ent in doc.ents:
            if ent.label_ in ['PER', 'PERSON']:
                nomes.append(ent.text)

        return nomes

    def contem_nomes(self, texto: str) -> bool:
        """
        Verifica se o texto contém nomes de pessoas.

        Args:
            texto: Texto a ser analisado

        Returns:
            True se contém nomes, False caso contrário
        """
        nomes = self.detectar_nomes(texto)
        return len(nomes) > 0


class ModeloHibridoDeteccao:
    """
    Modelo híbrido que combina regex, NER e machine learning
    para detectar dados pessoais em pedidos de acesso à informação.
    """

    def __init__(
        self,
        usar_regex: bool = True,
        usar_ner: bool = True,
        usar_ml: bool = True,
        threshold_confianca: float = 0.5
    ):
        """
        Inicializa o modelo híbrido.

        Args:
            usar_regex: Se True, usa detecção por regex
            usar_ner: Se True, usa detecção por NER
            usar_ml: Se True, usa classificador ML
            threshold_confianca: Limiar de confiança para classificação
        """
        self.usar_regex = usar_regex
        self.usar_ner = usar_ner
        self.usar_ml = usar_ml
        self.threshold_confianca = threshold_confianca

        # Inicializar componentes
        self.detector_regex = DetectorPadroesRegex() if usar_regex else None
        self.detector_ner = DetectorNER() if usar_ner else None

        # Componentes de ML
        self.vectorizer = None
        self.scaler = None
        self.modelo_ml = None
        self.treinado = False

    def extrair_features_ml(self, textos: pd.Series) -> np.ndarray:
        """
        Extrai features para o modelo de ML.

        Args:
            textos: Série com textos a serem processados

        Returns:
            Array com features extraídas
        """
        features = []

        for texto in textos:
            texto = str(texto) if not pd.isna(texto) else ""

            # Features numéricas
            feat = {
                'tamanho': len(texto),
                'qtd_numeros': len(re.findall(r'\d', texto)),
                'qtd_palavras': len(texto.split()),
                'tem_arroba': int('@' in texto),
                'tem_ponto': int('.' in texto),
                'tem_hifen': int('-' in texto),
                'tem_parenteses': int('(' in texto or ')' in texto),
                'razao_numeros': len(re.findall(r'\d', texto)) / max(len(texto), 1),
                'tem_seq_numeros': int(bool(re.search(r'\d{9,}', texto))),
            }

            features.append(list(feat.values()))

        return np.array(features)

    def treinar(
        self,
        df: pd.DataFrame,
        coluna_texto: str = 'texto_pedido',
        coluna_label: str = 'contem_dados_pessoais'
    ):
        """
        Treina o modelo híbrido.

        Args:
            df: DataFrame com dados de treino
            coluna_texto: Nome da coluna com o texto
            coluna_label: Nome da coluna com os rótulos
        """
        logger.info("Iniciando treinamento do modelo híbrido...")

        if not self.usar_ml:
            logger.info("ML desabilitado. Usando apenas regex e NER.")
            self.treinado = True
            return

        # Preparar dados
        X_texto = df[coluna_texto]
        y = df[coluna_label].astype(int)

        logger.info(f"Dados de treino: {len(X_texto)} amostras")
        logger.info(f"Distribuição de classes: {y.value_counts().to_dict()}")

        # TF-IDF
        logger.info("Criando features TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        X_tfidf = self.vectorizer.fit_transform(X_texto)

        # Features numéricas
        logger.info("Extraindo features numéricas...")
        X_numeric = self.extrair_features_ml(X_texto)

        # Normalizar features numéricas
        self.scaler = StandardScaler()
        X_numeric_scaled = self.scaler.fit_transform(X_numeric)

        # Combinar features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_numeric_scaled])

        # Treinar ensemble de modelos
        logger.info("Treinando modelos de ML...")

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Logistic Regression
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            C=1.0
        )

        # Ensemble (Voting)
        self.modelo_ml = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr)
            ],
            voting='soft',
            n_jobs=-1
        )

        self.modelo_ml.fit(X_combined, y)

        logger.info("Treinamento concluído!")
        self.treinado = True

    def prever_batch(
        self,
        textos: pd.Series,
        retornar_detalhes: bool = False
    ) -> np.ndarray:
        """
        Faz predições em um batch de textos.

        Args:
            textos: Série com textos a serem classificados
            retornar_detalhes: Se True, retorna detalhes das detecções

        Returns:
            Array com predições (True/False)
        """
        predicoes = []
        detalhes = [] if retornar_detalhes else None

        for texto in textos:
            pred, det = self.prever(texto, retornar_detalhes=retornar_detalhes)
            predicoes.append(pred)
            if retornar_detalhes:
                detalhes.append(det)

        if retornar_detalhes:
            return np.array(predicoes), detalhes
        return np.array(predicoes)

    def prever(
        self,
        texto: str,
        retornar_detalhes: bool = False
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Faz predição para um único texto.

        Args:
            texto: Texto a ser classificado
            retornar_detalhes: Se True, retorna detalhes das detecções

        Returns:
            Tupla (predição, detalhes) ou apenas predição
        """
        scores = {}

        # Camada 1: Regex
        if self.usar_regex:
            deteccoes_regex = self.detector_regex.detectar_todos(texto)
            score_regex = 1.0 if any(deteccoes_regex.values()) else 0.0
            scores['regex'] = score_regex
        else:
            deteccoes_regex = {}
            score_regex = 0.0

        # Camada 2: NER
        if self.usar_ner:
            contem_nomes = self.detector_ner.contem_nomes(texto)
            score_ner = 0.8 if contem_nomes else 0.0
            scores['ner'] = score_ner
        else:
            contem_nomes = False
            score_ner = 0.0

        # Camada 3: ML
        if self.usar_ml and self.treinado:
            # Preparar features
            X_tfidf = self.vectorizer.transform([texto])
            X_numeric = self.extrair_features_ml(pd.Series([texto]))
            X_numeric_scaled = self.scaler.transform(X_numeric)

            from scipy.sparse import hstack
            X_combined = hstack([X_tfidf, X_numeric_scaled])

            # Predição
            proba = self.modelo_ml.predict_proba(X_combined)[0]
            score_ml = proba[1]  # Probabilidade da classe positiva
            scores['ml'] = score_ml
        else:
            score_ml = 0.0

        # Combinar scores (estratégia de ensemble)
        # Se regex detecta algo óbvio (CPF, email, etc.), peso alto
        if score_regex > 0:
            score_final = 0.9  # Alta confiança
        # Se NER detecta nome e ML também indica, peso médio-alto
        elif score_ner > 0 and score_ml > 0.5:
            score_final = 0.8
        # Se apenas ML indica, usar score do ML
        elif score_ml > 0:
            score_final = score_ml
        # Se NER detecta nome mas ML não confirma, peso médio
        elif score_ner > 0:
            score_final = 0.6
        else:
            score_final = 0.0

        # Decisão final
        predicao = score_final >= self.threshold_confianca

        if retornar_detalhes:
            detalhes = {
                'scores': scores,
                'score_final': score_final,
                'deteccoes_regex': deteccoes_regex,
                'contem_nomes': contem_nomes,
                'predicao': predicao
            }
            return predicao, detalhes

        return predicao, None

    def salvar(self, caminho: str):
        """
        Salva o modelo treinado.

        Args:
            caminho: Caminho do arquivo onde salvar o modelo
        """
        logger.info(f"Salvando modelo em: {caminho}")

        modelo_dict = {
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'modelo_ml': self.modelo_ml,
            'usar_regex': self.usar_regex,
            'usar_ner': self.usar_ner,
            'usar_ml': self.usar_ml,
            'threshold_confianca': self.threshold_confianca,
            'treinado': self.treinado
        }

        # Criar diretório se não existir
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(modelo_dict, caminho)
        logger.info("Modelo salvo com sucesso!")

    @classmethod
    def carregar(cls, caminho: str):
        """
        Carrega um modelo salvo.

        Args:
            caminho: Caminho do arquivo do modelo

        Returns:
            Instância do modelo carregado
        """
        logger.info(f"Carregando modelo de: {caminho}")

        modelo_dict = joblib.load(caminho)

        # Criar instância
        modelo = cls(
            usar_regex=modelo_dict['usar_regex'],
            usar_ner=modelo_dict['usar_ner'],
            usar_ml=modelo_dict['usar_ml'],
            threshold_confianca=modelo_dict['threshold_confianca']
        )

        # Restaurar componentes
        modelo.vectorizer = modelo_dict['vectorizer']
        modelo.scaler = modelo_dict['scaler']
        modelo.modelo_ml = modelo_dict['modelo_ml']
        modelo.treinado = modelo_dict['treinado']

        logger.info("Modelo carregado com sucesso!")
        return modelo


# Testes
if __name__ == "__main__":
    print("Módulo de modelo carregado com sucesso!")

    # Teste rápido de regex
    detector = DetectorPadroesRegex()

    textos_teste = [
        "Meu CPF é 123.456.789-00",
        "Entre em contato: joao@email.com",
        "Telefone: (61) 98765-4321",
        "Solicito informações sobre o orçamento",
    ]

    print("\nTeste de detecção por regex:")
    for texto in textos_teste:
        resultado = detector.contem_dados_pessoais(texto)
        print(f"'{texto}' -> {resultado}")
