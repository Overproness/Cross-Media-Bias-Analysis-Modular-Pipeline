"""
Component 2: Lexical Choice Extraction
Identifies how different sources describe the same concepts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class LexicalExtractor:
    """Base class for lexical choice extraction"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract lexical choices per source"""
        raise NotImplementedError


class TFIDFKeywordExtractor(LexicalExtractor):
    """Extract keywords using TF-IDF"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf_config = config.get('tfidf_keywords', {})
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 100),
            min_df=tfidf_config.get('min_df', 2),
            max_df=tfidf_config.get('max_df', 0.8),
            ngram_range=(1, 2)
        )
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Dict]:
        logger.info("Extracting keywords using TF-IDF...")
        
        results = {}
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]['article'].tolist()
            source_text = ' '.join(source_articles)
            
            # Fit TF-IDF on all sources to get common vocabulary
            all_text = df.groupby('source')['article'].apply(' '.join).tolist()
            self.vectorizer.fit(all_text)
            
            # Transform this source
            tfidf_matrix = self.vectorizer.transform([source_text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-50:][::-1]
            
            keywords = {
                feature_names[idx]: float(scores[idx])
                for idx in top_indices if scores[idx] > 0
            }
            
            results[source] = {
                'keywords': keywords,
                'top_20': list(keywords.keys())[:20]
            }
        
        return results


class RAKEExtractor(LexicalExtractor):
    """Extract keywords using RAKE algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from rake_nltk import Rake
            rake_config = config.get('rake', {})
            self.rake = Rake(
                max_length=rake_config.get('max_words', 4),
                min_frequency=rake_config.get('min_freq', 2)
            )
        except ImportError:
            logger.error("rake-nltk not installed")
            self.rake = None
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Dict]:
        if not self.rake:
            raise ImportError("rake-nltk not available")
        
        logger.info("Extracting keywords using RAKE...")
        
        results = {}
        
        for source in df['source'].unique():
            source_text = ' '.join(df[df['source'] == source]['article'].tolist())
            
            self.rake.extract_keywords_from_text(source_text)
            keywords_with_scores = self.rake.get_ranked_phrases_with_scores()
            
            keywords = {
                phrase: score
                for score, phrase in keywords_with_scores[:50]
            }
            
            results[source] = {
                'keywords': keywords,
                'top_20': list(keywords.keys())[:20]
            }
        
        return results


class YAKEExtractor(LexicalExtractor):
    """Extract keywords using YAKE algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            import yake
            yake_config = config.get('yake', {})
            self.extractor = yake.KeywordExtractor(
                lan="en",
                n=yake_config.get('max_ngram_size', 3),
                dedupLim=0.9,
                top=yake_config.get('num_keywords', 20)
            )
        except ImportError:
            logger.error("yake not installed")
            self.extractor = None
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Dict]:
        if not self.extractor:
            raise ImportError("yake not available")
        
        logger.info("Extracting keywords using YAKE...")
        
        results = {}
        
        for source in df['source'].unique():
            source_text = ' '.join(df[df['source'] == source]['article'].tolist())
            
            keywords_with_scores = self.extractor.extract_keywords(source_text)
            
            keywords = {
                phrase: score
                for phrase, score in keywords_with_scores
            }
            
            results[source] = {
                'keywords': keywords,
                'top_20': list(keywords.keys())[:20]
            }
        
        return results


class POSFilteredExtractor(LexicalExtractor):
    """Extract terms filtered by POS tags"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
            pos_config = config.get('pos_filtered', {})
            self.allowed_pos = pos_config.get('allowed_pos', ['NOUN', 'VERB', 'ADJ'])
            self.min_freq = pos_config.get('min_freq', 3)
        except ImportError:
            self.nlp = None
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Dict]:
        if not self.nlp:
            raise ImportError("spacy not available")
        
        logger.info("Extracting POS-filtered terms...")
        
        results = {}
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]['article'].tolist()
            
            terms = []
            for article in source_articles[:100]:  # Limit for performance
                doc = self.nlp(article[:10000])  # Limit text length
                for token in doc:
                    if token.pos_ in self.allowed_pos and not token.is_stop:
                        terms.append(token.lemma_.lower())
            
            # Count frequencies
            term_counts = Counter(terms)
            
            # Filter by frequency
            keywords = {
                term: count
                for term, count in term_counts.most_common(50)
                if count >= self.min_freq
            }
            
            results[source] = {
                'keywords': keywords,
                'top_20': list(keywords.keys())[:20]
            }
        
        return results


class ContextualWordDetector(LexicalExtractor):
    """Detect words used in similar contexts across sources"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.cw_config = config.get('contextual_words', {})
        try:
            from gensim.models import Word2Vec
            self.Word2Vec = Word2Vec
            self.gensim_available = True
        except ImportError:
            logger.warning("gensim not installed, ContextualWordDetector will not function")
            self.Word2Vec = None
            self.gensim_available = False
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Dict]:
        if not self.gensim_available:
            logger.error("Cannot extract contextual words: gensim not installed. Install with: pip install gensim")
            return {}
        
        logger.info("Detecting contextual word usage...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Train Word2Vec per source
        source_models = {}
        
        for source in df['source'].unique():
            articles = df[df['source'] == source]['article'].tolist()
            sentences = [article.split() for article in articles]
            
            model = self.Word2Vec(
                sentences,
                vector_size=100,
                window=self.cw_config.get('window_size', 5),
                min_count=self.cw_config.get('min_count', 5),
                workers=4
            )
            source_models[source] = model
        
        # Find contextually similar words
        results = {}
        min_similarity = self.cw_config.get('min_similarity', 0.6)
        
        for source, model in source_models.items():
            vocab = list(model.wv.index_to_key)[:200]
            
            contextual_groups = defaultdict(list)
            
            for word in vocab:
                similar = model.wv.most_similar(word, topn=5)
                contextual_groups[word] = [w for w, sim in similar if sim >= min_similarity]
            
            results[source] = {
                'contextual_groups': dict(contextual_groups),
                'vocabulary_size': len(vocab)
            }
        
        return results


class SynonymDetector(LexicalExtractor):
    """Detect synonyms and semantically similar terms across sources"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.syn_config = config.get('synonym_detection', {})
        self.similarity_threshold = self.syn_config.get('similarity_threshold', 0.7)
        self.use_wordnet = self.syn_config.get('use_wordnet', True)
        
        # Try to load WordNet
        if self.use_wordnet:
            try:
                from nltk.corpus import wordnet
                self.wordnet = wordnet
            except ImportError:
                logger.warning("NLTK WordNet not available, using semantic similarity only")
                self.wordnet = None
        else:
            self.wordnet = None
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Dict]:
        logger.info("Detecting synonyms and semantically similar terms...")
        
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            
            results = {}
            
            for source in df['source'].unique():
                # Get top terms from TF-IDF
                source_articles = df[df['source'] == source]['article'].tolist()
                
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(source_articles)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get embeddings for top terms
                term_embeddings = model.encode(list(feature_names))
                
                # Find similar terms
                similarity_matrix = cosine_similarity(term_embeddings)
                
                synonym_groups = defaultdict(list)
                for i, term in enumerate(feature_names):
                    for j, other_term in enumerate(feature_names):
                        if i != j and similarity_matrix[i, j] >= self.similarity_threshold:
                            synonym_groups[term].append((other_term, float(similarity_matrix[i, j])))
                
                # Sort by similarity
                for term in synonym_groups:
                    synonym_groups[term] = sorted(synonym_groups[term], key=lambda x: x[1], reverse=True)[:5]
                
                results[source] = {
                    'synonym_groups': dict(synonym_groups),
                    'total_terms': len(feature_names)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Synonym detection failed: {e}")
            return {}


class EntityLabelingExtractor(LexicalExtractor):
    """Extract and compare entity descriptors/labels across sources"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.entity_config = config.get('entity_labeling', {})
        self.compare_descriptors = self.entity_config.get('compare_descriptors', True)
        self.min_descriptor_freq = self.entity_config.get('min_descriptor_freq', 2)
        
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except ImportError:
            logger.error("spacy not installed")
            self.nlp = None
        except OSError:
            logger.error("spacy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Dict]:
        if not self.nlp:
            logger.error("Cannot extract entity labels: spacy not available")
            return {}
        
        logger.info("Extracting entity labels and descriptors...")
        
        results = {}
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]['article'].tolist()
            
            entity_descriptors = defaultdict(list)
            entity_counts = Counter()
            
            # Process a sample of articles for performance
            sample_size = min(100, len(source_articles))
            for article in source_articles[:sample_size]:
                # Limit article length
                doc = self.nlp(article[:10000])
                
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                        entity_counts[ent.text] += 1
                        
                        # Extract surrounding adjectives/descriptors
                        for token in ent.root.head.children:
                            if token.pos_ == 'ADJ':
                                entity_descriptors[ent.text].append(token.text.lower())
            
            # Get top entities
            top_entities = [ent for ent, _ in entity_counts.most_common(20)]
            
            # Count descriptor frequencies
            descriptor_summary = {}
            for entity in top_entities:
                if entity in entity_descriptors:
                    desc_counts = Counter(entity_descriptors[entity])
                    filtered = {d: c for d, c in desc_counts.items() if c >= self.min_descriptor_freq}
                    if filtered:
                        descriptor_summary[entity] = filtered
            
            results[source] = {
                'top_entities': top_entities,
                'entity_descriptors': descriptor_summary,
                'total_entities': len(entity_counts)
            }
        
        return results


class LexicalExtractionFactory:
    """Factory for creating lexical extractors"""
    
    @staticmethod
    def create(config: Dict, methods: List[str]) -> List[LexicalExtractor]:
        """
        Create lexical extractors based on configuration
        
        Args:
            config: Configuration dictionary
            methods: List of method names
            
        Returns:
            List of LexicalExtractor instances
        """
        extractors_map = {
            'tfidf_keywords': TFIDFKeywordExtractor,
            'rake': RAKEExtractor,
            'yake': YAKEExtractor,
            'pos_filtered': POSFilteredExtractor,
            'contextual_words': ContextualWordDetector,
            'synonym_detection': SynonymDetector,
            'entity_labeling': EntityLabelingExtractor
        }
        
        extractors = []
        for method in methods:
            if method in extractors_map:
                extractors.append(extractors_map[method](config))
            else:
                logger.warning(f"Unknown lexical extraction method: {method}")
        
        return extractors


def compare_lexical_choices(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare lexical choices across sources
    
    Args:
        results: Dictionary with results per source
        
    Returns:
        DataFrame with comparison metrics
    """
    sources = list(results.keys())
    
    # Create comparison matrix
    comparison_data = []
    
    for source in sources:
        keywords = set(results[source]['top_20'])
        
        for other_source in sources:
            if source != other_source:
                other_keywords = set(results[other_source]['top_20'])
                
                overlap = len(keywords & other_keywords)
                unique = len(keywords - other_keywords)
                
                comparison_data.append({
                    'source': source,
                    'compared_to': other_source,
                    'overlap': overlap,
                    'unique_terms': unique,
                    'jaccard_similarity': overlap / len(keywords | other_keywords) if keywords | other_keywords else 0
                })
    
    return pd.DataFrame(comparison_data)
