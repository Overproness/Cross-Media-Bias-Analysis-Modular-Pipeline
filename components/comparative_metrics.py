"""
Component 8: Comparative Metrics
Quantifies discovered patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Base class for comparative metrics"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Calculate metrics"""
        raise NotImplementedError


class BiasScoreCalculator(MetricCalculator):
    """Calculate aggregate bias scores"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        bs_config = config.get('bias_score', {})
        self.components = bs_config.get('components', ['sentiment', 'framing', 'lexical'])
        self.weights = bs_config.get('weights', [0.4, 0.4, 0.2])
    
    def calculate(self, df: pd.DataFrame, lexical_data: Dict = None, framing_data: Dict = None) -> Dict:
        logger.info("Calculating bias scores...")
        
        bias_scores = {}
        
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            
            component_scores = {}
            
            # Sentiment bias (deviation from neutral)
            if 'sentiment' in self.components and 'sentiment_compound' in source_df.columns:
                avg_sentiment = source_df['sentiment_compound'].mean()
                sentiment_bias = abs(avg_sentiment)  # Distance from neutral (0)
                component_scores['sentiment'] = sentiment_bias
            
            # Framing bias (frame diversity)
            if 'framing' in self.components and 'dominant_frame' in source_df.columns:
                frame_dist = source_df['dominant_frame'].value_counts(normalize=True)
                # Higher entropy = less biased (more diverse framing)
                entropy = stats.entropy(frame_dist.values)
                max_entropy = np.log(len(frame_dist))
                framing_bias = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
                component_scores['framing'] = framing_bias
            
            # Lexical bias (uniqueness of vocabulary)
            if 'lexical' in self.components and lexical_data:
                if source in lexical_data:
                    # Compare with other sources
                    other_sources = [s for s in lexical_data.keys() if s != source]
                    
                    if other_sources:
                        source_keywords = set(lexical_data[source]['top_20'])
                        
                        overlaps = []
                        for other in other_sources:
                            other_keywords = set(lexical_data[other]['top_20'])
                            overlap = len(source_keywords & other_keywords) / 20
                            overlaps.append(overlap)
                        
                        # Lower overlap = higher bias
                        lexical_bias = 1 - np.mean(overlaps)
                        component_scores['lexical'] = lexical_bias
            
            # Weighted combination
            if component_scores:
                available_components = list(component_scores.keys())
                available_weights = [self.weights[self.components.index(c)] 
                                    for c in available_components if c in self.components]
                
                # Normalize weights
                if sum(available_weights) > 0:
                    normalized_weights = np.array(available_weights) / sum(available_weights)
                    
                    overall_bias = sum(
                        component_scores[c] * w 
                        for c, w in zip(available_components, normalized_weights)
                    )
                else:
                    overall_bias = 0
                
                bias_scores[source] = {
                    'overall_bias': float(overall_bias),
                    'component_scores': {k: float(v) for k, v in component_scores.items()}
                }
        
        return bias_scores


class FramingDivergenceCalculator(MetricCalculator):
    """Calculate framing divergence index"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        fd_config = config.get('framing_divergence', {})
        self.method = fd_config.get('method', 'jensen_shannon')
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        logger.info("Calculating framing divergence...")
        
        if 'dominant_frame' not in df.columns:
            logger.warning("No framing information available")
            return {}
        
        # Get frame distributions per source
        sources = df['source'].unique()
        frame_distributions = {}
        
        all_frames = df['dominant_frame'].unique()
        
        for source in sources:
            source_df = df[df['source'] == source]
            frame_counts = source_df['dominant_frame'].value_counts()
            
            # Create full distribution with all frames
            distribution = []
            for frame in all_frames:
                count = frame_counts.get(frame, 0)
                distribution.append(count)
            
            # Normalize
            total = sum(distribution)
            distribution = [d/total if total > 0 else 0 for d in distribution]
            frame_distributions[source] = distribution
        
        # Calculate pairwise divergence
        divergence_matrix = {}
        
        for i, source1 in enumerate(sources):
            divergence_matrix[source1] = {}
            
            for source2 in sources:
                if source1 == source2:
                    divergence_matrix[source1][source2] = 0.0
                    continue
                
                dist1 = np.array(frame_distributions[source1])
                dist2 = np.array(frame_distributions[source2])
                
                if self.method == 'jensen_shannon':
                    # Add small epsilon to avoid log(0)
                    dist1 = dist1 + 1e-10
                    dist2 = dist2 + 1e-10
                    divergence = jensenshannon(dist1, dist2)
                
                elif self.method == 'kl_divergence':
                    dist1 = dist1 + 1e-10
                    dist2 = dist2 + 1e-10
                    divergence = stats.entropy(dist1, dist2)
                
                else:  # hellinger
                    divergence = np.sqrt(0.5 * np.sum((np.sqrt(dist1) - np.sqrt(dist2))**2))
                
                divergence_matrix[source1][source2] = float(divergence)
        
        return {
            'divergence_matrix': divergence_matrix,
            'method': self.method
        }


class LexicalDiversityCalculator(MetricCalculator):
    """Calculate lexical diversity measures"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        ld_config = config.get('lexical_diversity', {})
        self.measures = ld_config.get('measures', ['ttr', 'mtld'])
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        logger.info("Calculating lexical diversity...")
        
        diversity_scores = {}
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]['article'].tolist()
            all_text = ' '.join(source_articles)
            tokens = all_text.lower().split()
            
            scores = {}
            
            # Type-Token Ratio (TTR)
            if 'ttr' in self.measures:
                types = len(set(tokens))
                tokens_count = len(tokens)
                ttr = types / tokens_count if tokens_count > 0 else 0
                scores['ttr'] = float(ttr)
            
            # MTLD (Measure of Textual Lexical Diversity) - simplified version
            if 'mtld' in self.measures:
                mtld = self._calculate_mtld(tokens)
                scores['mtld'] = float(mtld)
            
            # Unique words per 100 words
            scores['unique_per_100'] = float((len(set(tokens)) / len(tokens)) * 100 if tokens else 0)
            
            diversity_scores[source] = scores
        
        return diversity_scores
    
    def _calculate_mtld(self, tokens: List[str], threshold: float = 0.72) -> float:
        """Calculate MTLD (simplified)"""
        if not tokens:
            return 0
        
        factor = 0
        types_seen = set()
        current_ttr = 1.0
        
        for token in tokens:
            types_seen.add(token)
            current_ttr = len(types_seen) / (len(types_seen) + factor)
            
            if current_ttr < threshold:
                factor += 1
                types_seen = set()
        
        return len(tokens) / max(1, factor)


class ObjectivityCalculator(MetricCalculator):
    """Calculate objectivity metrics"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.obj_config = config.get('objectivity_score', {})
        
        # Hedge words indicating uncertainty/subjectivity
        self.hedge_words = [
            'perhaps', 'maybe', 'possibly', 'probably', 'might', 'may', 'could',
            'seems', 'appears', 'suggests', 'indicates', 'allegedly', 'reportedly'
        ]
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        logger.info("Calculating objectivity scores...")
        
        objectivity_scores = {}
        
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            
            scores = {}
            
            # Hedge word usage
            if self.obj_config.get('hedge_words', True):
                hedge_counts = []
                
                for article in source_df['article']:
                    article_lower = article.lower()
                    count = sum(1 for word in self.hedge_words if word in article_lower)
                    word_count = len(article.split())
                    hedge_counts.append(count / word_count if word_count > 0 else 0)
                
                scores['hedge_word_density'] = float(np.mean(hedge_counts))
            
            # Subjectivity (if available from TextBlob)
            if 'sentiment_subjectivity' in source_df.columns:
                scores['avg_subjectivity'] = float(source_df['sentiment_subjectivity'].mean())
                scores['objectivity'] = float(1 - scores['avg_subjectivity'])
            
            # Combined objectivity score
            if 'hedge_word_density' in scores and 'objectivity' in scores:
                # Lower hedge words and lower subjectivity = higher objectivity
                objectivity = (scores['objectivity'] + (1 - scores['hedge_word_density'] * 10)) / 2
                objectivity = max(0, min(1, objectivity))  # Clip to [0, 1]
            elif 'objectivity' in scores:
                objectivity = scores['objectivity']
            else:
                objectivity = 0.5
            
            scores['overall_objectivity'] = float(objectivity)
            
            objectivity_scores[source] = scores
        
        return objectivity_scores


class ComparativeMetricsFactory:
    """Factory for creating metric calculators"""
    
    @staticmethod
    def create(config: Dict, metrics: List[str]) -> List[MetricCalculator]:
        """
        Create metric calculators based on configuration
        
        Args:
            config: Configuration dictionary
            metrics: List of metric names
            
        Returns:
            List of MetricCalculator instances
        """
        calculators_map = {
            'bias_score': BiasScoreCalculator,
            'framing_divergence': FramingDivergenceCalculator,
            'lexical_diversity': LexicalDiversityCalculator,
            'objectivity_score': ObjectivityCalculator,
            'political_lean': BiasScoreCalculator  # Use BiasScoreCalculator as proxy for political lean
        }
        
        calculators = []
        for metric in metrics:
            if metric in calculators_map:
                calculators.append(calculators_map[metric](config))
            else:
                logger.warning(f"Unknown comparative metric: {metric}")
        
        return calculators
