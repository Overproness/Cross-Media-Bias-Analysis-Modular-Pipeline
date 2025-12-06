"""
Component 1: Event Detection & Clustering
Groups articles covering the same event/topic using various methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class EventDetector:
    """Base class for event detection methods"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect events and assign cluster labels
        
        Args:
            df: DataFrame with articles
            
        Returns:
            DataFrame with cluster labels
        """
        raise NotImplementedError


class LDAEventDetector(EventDetector):
    """Topic modeling using Latent Dirichlet Allocation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        
        self.lda_config = config.get('lda', {})
        self.vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.8)
        self.lda = LatentDirichletAllocation(
            n_components=self.lda_config.get('n_topics', 20),
            max_iter=self.lda_config.get('max_iter', 100),
            random_state=self.lda_config.get('random_state', 42)
        )
    
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Detecting events using LDA...")
        
        # Vectorize documents
        doc_term_matrix = self.vectorizer.fit_transform(df['article'])
        
        # Fit LDA
        topic_distributions = self.lda.fit_transform(doc_term_matrix)
        
        # Assign dominant topic
        df['cluster'] = topic_distributions.argmax(axis=1)
        df['cluster_probability'] = topic_distributions.max(axis=1)
        
        # Get topic keywords
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        for idx, topic in enumerate(self.lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            topics[idx] = [feature_names[i] for i in top_indices]
        
        df['topic_keywords'] = df['cluster'].map(topics)
        
        logger.info(f"Detected {len(topics)} topics")
        return df


class NMFEventDetector(EventDetector):
    """Topic modeling using Non-negative Matrix Factorization"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.nmf_config = config.get('nmf', {})
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
        self.nmf = NMF(
            n_components=self.nmf_config.get('n_topics', 20),
            max_iter=self.nmf_config.get('max_iter', 200),
            random_state=self.nmf_config.get('random_state', 42)
        )
    
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Detecting events using NMF...")
        
        # Vectorize documents
        tfidf_matrix = self.vectorizer.fit_transform(df['article'])
        
        # Fit NMF
        topic_distributions = self.nmf.fit_transform(tfidf_matrix)
        
        # Assign dominant topic
        df['cluster'] = topic_distributions.argmax(axis=1)
        df['cluster_probability'] = topic_distributions.max(axis=1)
        
        # Get topic keywords
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        for idx, topic in enumerate(self.nmf.components_):
            top_indices = topic.argsort()[-10:][::-1]
            topics[idx] = [feature_names[i] for i in top_indices]
        
        df['topic_keywords'] = df['cluster'].map(topics)
        
        logger.info(f"Detected {len(topics)} topics")
        return df


class BERTopicEventDetector(EventDetector):
    """Topic modeling using BERTopic"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            import torch
            
            self.bertopic_config = config.get('bertopic', {})
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create embedding model with GPU support
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            
            self.model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=self.bertopic_config.get('min_topic_size', 10),
                nr_topics=self.bertopic_config.get('nr_topics', 'auto'),
                calculate_probabilities=self.bertopic_config.get('calculate_probabilities', True)
            )
        except ImportError:
            logger.error("BERTopic not installed")
            self.model = None
    
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model:
            raise ImportError("BERTopic not available")
        
        logger.info("Detecting events using BERTopic...")
        
        documents = df['article'].tolist()
        topics, probabilities = self.model.fit_transform(documents)
        
        df['cluster'] = topics
        df['cluster_probability'] = probabilities.max(axis=1) if len(probabilities.shape) > 1 else probabilities
        
        # Get topic keywords
        topic_info = self.model.get_topic_info()
        logger.info(f"Detected {len(topic_info)} topics")
        
        return df


class SemanticClusteringDetector(EventDetector):
    """Clustering using sentence embeddings"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            self.clustering_config = config.get('semantic_clustering', {})
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if torch.cuda.is_available():
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                logger.info("No GPU detected, using CPU")
            
            self.model = SentenceTransformer(
                self.clustering_config.get('embedding_model', 'all-MiniLM-L6-v2'),
                device=device
            )
            self.batch_size = self.clustering_config.get('batch_size', 64 if device == 'cuda' else 32)
        except ImportError:
            logger.error("sentence-transformers not installed")
            self.model = None
    
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model:
            raise ImportError("sentence-transformers not available")
        
        logger.info("Detecting events using semantic clustering...")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            df['article'].tolist(), 
            show_progress_bar=True,
            batch_size=self.batch_size
        )
        
        # Clustering
        algorithm = self.clustering_config.get('clustering_algorithm', 'kmeans')
        
        if algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            n_clusters = self.clustering_config.get('n_clusters', 20)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(embeddings)
            
        elif algorithm == 'hdbscan':
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
                labels = clusterer.fit_predict(embeddings)
            except ImportError:
                logger.error("hdbscan not installed, falling back to KMeans")
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=20, random_state=42)
                labels = clusterer.fit_predict(embeddings)
        
        elif algorithm == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = self.clustering_config.get('n_clusters', 20)
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(embeddings)
        
        df['cluster'] = labels
        df['embedding'] = list(embeddings)
        
        logger.info(f"Detected {len(set(labels))} clusters")
        return df


class TemporalSemanticDetector(EventDetector):
    """Clustering combining temporal and semantic similarity"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            self.config_ts = config.get('temporal_semantic', {})
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except ImportError:
            self.model = None
    
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model:
            raise ImportError("sentence-transformers not available")
        
        logger.info("Detecting events using temporal-semantic clustering...")
        
        # Generate embeddings with batching and progress bar
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            df['article'].tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        
        # For large datasets, use a more memory-efficient approach
        # Instead of full similarity matrix, use mini-batch KMeans
        dates = pd.to_datetime(df['date'])
        time_window = self.config_ts.get('time_window_days', 7)
        n_clusters = self.config_ts.get('n_clusters', 20)
        
        # Add temporal features to embeddings
        # Normalize dates to 0-1 range
        date_numeric = (dates - dates.min()).dt.days.values
        date_normalized = date_numeric / (date_numeric.max() + 1e-8)
        
        # Create augmented feature matrix
        semantic_weight = self.config_ts.get('semantic_weight', 0.7)
        temporal_weight = self.config_ts.get('temporal_weight', 0.3)
        
        # Scale temporal feature to match semantic embedding magnitude
        temporal_feature = date_normalized.reshape(-1, 1) * temporal_weight * np.linalg.norm(embeddings[0])
        semantic_features = embeddings * semantic_weight
        
        # Combine features
        combined_features = np.concatenate([semantic_features, temporal_feature], axis=1)
        
        logger.info(f"Clustering {len(df)} articles into {n_clusters} clusters...")
        
        # Use MiniBatch KMeans for memory efficiency
        from sklearn.cluster import MiniBatchKMeans
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=1000,
            max_iter=100,
            verbose=0
        )
        labels = clusterer.fit_predict(combined_features)
        
        df['cluster'] = labels
        
        logger.info(f"Detected {len(set(labels))} clusters")
        return df


class HeadlineSimilarityDetector(EventDetector):
    """Clustering based on headline similarity"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.hs_config = config.get('headline_similarity', {})
    
    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Detecting events using headline similarity...")
        
        method = self.hs_config.get('method', 'tfidf')
        threshold = self.hs_config.get('threshold', 0.6)
        
        if method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(df['headline'])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
        else:  # embedding
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(df['headline'].tolist())
            similarity_matrix = cosine_similarity(embeddings)
        
        # Create adjacency matrix based on threshold
        adjacency = (similarity_matrix >= threshold).astype(int)
        
        # Find connected components (clusters)
        from scipy.sparse.csgraph import connected_components
        n_clusters, labels = connected_components(adjacency, directed=False)
        
        df['cluster'] = labels
        
        logger.info(f"Detected {n_clusters} clusters")
        return df


class EventDetectionFactory:
    """Factory for creating event detection instances"""
    
    @staticmethod
    def create(config: Dict) -> EventDetector:
        """
        Create event detector based on configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            EventDetector instance
        """
        method = config.get('method', 'semantic_clustering')
        
        detectors = {
            'lda': LDAEventDetector,
            'nmf': NMFEventDetector,
            'bertopic': BERTopicEventDetector,
            'semantic_clustering': SemanticClusteringDetector,
            'temporal_semantic': TemporalSemanticDetector,
            'headline_similarity': HeadlineSimilarityDetector
        }
        
        if method not in detectors:
            raise ValueError(f"Unknown detection method: {method}")
        
        return detectors[method](config)
