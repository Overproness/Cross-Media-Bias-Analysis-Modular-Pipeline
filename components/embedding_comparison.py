"""
Component 5: Embedding-based Comparison
Measures semantic distances between sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbeddingComparator:
    """Base class for embedding-based comparison"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def compare(self, df: pd.DataFrame) -> Dict:
        """Compare sources using embeddings"""
        raise NotImplementedError


class ContextualizedEmbeddingComparator(EmbeddingComparator):
    """Compare using contextualized embeddings (BERT, RoBERTa)"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            ce_config = config.get('contextualized_embeddings', {})
            model_name = ce_config.get('model_name', 'bert-base-uncased')
            
            # GPU detection
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.pooling = ce_config.get('pooling', 'mean')
            self.max_length = ce_config.get('max_length', 512)
            self.torch = torch
            
            logger.info(f"Using device: {self.device} for contextualized embeddings")
        except ImportError:
            logger.error("transformers or torch not installed")
            self.model = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        if self.pooling == 'cls':
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif self.pooling == 'mean':
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        else:  # max
            embedding = outputs.last_hidden_state.max(dim=1)[0].cpu().numpy()
        
        return embedding.flatten()
    
    def compare(self, df: pd.DataFrame) -> Dict:
        if not self.model:
            raise ImportError("transformers not available")
        
        logger.info("Comparing sources using contextualized embeddings...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get embeddings per source
        source_embeddings = {}
        
        for source in df['source'].unique():
            # Get a sample of articles directly from the dataframe
            source_df = df[df['source'] == source]
            sample_size = min(50, len(source_df))
            sampled_df = source_df.sample(n=sample_size, random_state=42)
            
            embeddings = []
            # Process only first 10 articles to limit memory usage
            for idx, article in enumerate(sampled_df['article'].head(10)):
                try:
                    # Truncate article to first 512 characters
                    text = str(article)[:512] if article else ""
                    if text.strip():
                        emb = self._get_embedding(text)
                        embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to get embedding for article in {source}: {e}")
                    continue
            
            if not embeddings:
                logger.warning(f"No embeddings generated for {source}, using zero vector")
                source_embeddings[source] = np.zeros(768)  # Default embedding size
            else:
                # Average embedding for source
                source_embeddings[source] = np.mean(embeddings, axis=0)
        
        # Compute pairwise similarities
        sources = list(source_embeddings.keys())
        similarity_matrix = np.zeros((len(sources), len(sources)))
        
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                sim = cosine_similarity(
                    source_embeddings[source1].reshape(1, -1),
                    source_embeddings[source2].reshape(1, -1)
                )[0, 0]
                similarity_matrix[i, j] = sim
        
        return {
            'sources': sources,
            'similarity_matrix': similarity_matrix.tolist(),
            'source_embeddings': {k: v.tolist() for k, v in source_embeddings.items()}
        }


class WordEmbeddingShiftComparator(EmbeddingComparator):
    """Compare word embedding spaces trained per source"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        we_config = config.get('word_embedding_shifts', {})
        self.embedding_type = we_config.get('embedding_type', 'word2vec')
        self.vector_size = we_config.get('vector_size', 100)
        self.window = we_config.get('window', 5)
        self.min_count = we_config.get('min_count', 5)
    
    def compare(self, df: pd.DataFrame) -> Dict:
        logger.info("Comparing word embedding shifts across sources...")
        
        from gensim.models import Word2Vec
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Train Word2Vec per source
        source_models = {}
        
        for source in df['source'].unique():
            articles = df[df['source'] == source]['article'].tolist()
            sentences = [article.lower().split() for article in articles]
            
            model = Word2Vec(
                sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=4
            )
            source_models[source] = model
        
        # Find common vocabulary
        common_vocab = set.intersection(*[set(model.wv.index_to_key) for model in source_models.values()])
        common_vocab = list(common_vocab)[:100]  # Limit
        
        logger.info(f"Common vocabulary size: {len(common_vocab)}")
        
        # Compare word vectors across sources
        word_shifts = {}
        
        for word in common_vocab[:20]:  # Sample words
            shifts = {}
            sources = list(source_models.keys())
            
            for i, source1 in enumerate(sources):
                for source2 in sources[i+1:]:
                    vec1 = source_models[source1].wv[word]
                    vec2 = source_models[source2].wv[word]
                    
                    similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
                    shifts[f"{source1}_vs_{source2}"] = 1 - similarity  # Convert to distance
            
            word_shifts[word] = shifts
        
        return {
            'source_models': source_models,
            'common_vocab': common_vocab,
            'word_shifts': word_shifts
        }


class SemanticDivergenceComparator(EmbeddingComparator):
    """Measure semantic divergence between source pairs"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except ImportError:
            self.model = None
    
    def compare(self, df: pd.DataFrame) -> Dict:
        if not self.model:
            raise ImportError("sentence-transformers not available")
        
        logger.info("Measuring semantic divergence between sources...")
        
        from scipy.spatial.distance import jensenshannon
        from sklearn.metrics.pairwise import cosine_distances
        
        # Get embeddings for all articles
        if 'embedding' not in df.columns:
            logger.info("Generating embeddings...")
            embeddings = self.model.encode(df['article'].tolist(), show_progress_bar=True)
            df['embedding'] = list(embeddings)
        
        # Compute divergence between sources
        sources = df['source'].unique()
        divergence_matrix = np.zeros((len(sources), len(sources)))
        
        for i, source1 in enumerate(sources):
            emb1 = np.array(df[df['source'] == source1]['embedding'].tolist())
            
            for j, source2 in enumerate(sources):
                if i == j:
                    continue
                
                emb2 = np.array(df[df['source'] == source2]['embedding'].tolist())
                
                # Compute average distance
                distances = cosine_distances(emb1[:50], emb2[:50])  # Sample
                avg_distance = distances.mean()
                
                divergence_matrix[i, j] = avg_distance
        
        return {
            'sources': list(sources),
            'divergence_matrix': divergence_matrix
        }


class EmbeddingComparisonFactory:
    """Factory for creating embedding comparators"""
    
    @staticmethod
    def create(config: Dict, methods: List[str]) -> List[EmbeddingComparator]:
        """
        Create embedding comparators based on configuration
        
        Args:
            config: Configuration dictionary
            methods: List of method names
            
        Returns:
            List of EmbeddingComparator instances
        """
        comparators_map = {
            'contextualized_embeddings': ContextualizedEmbeddingComparator,
            'word_embedding_shifts': WordEmbeddingShiftComparator,
            'semantic_change': SemanticDivergenceComparator,
            'contrastive_analysis': SemanticDivergenceComparator
        }
        
        comparators = []
        for method in methods:
            if method in comparators_map:
                comparators.append(comparators_map[method](config))
            else:
                logger.warning(f"Unknown embedding comparison method: {method}")
        
        return comparators
