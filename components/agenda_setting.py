"""
Component 6: Agenda-Setting Analysis
Quantifies topic emphasis and coverage patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class AgendaAnalyzer:
    """Base class for agenda-setting analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze agenda-setting patterns"""
        raise NotImplementedError


class TopicFrequencyAnalyzer(AgendaAnalyzer):
    """Analyze topic frequency over time"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        tf_config = config.get('topic_frequency', {})
        self.time_granularity = tf_config.get('time_granularity', 'week')
        self.normalize = tf_config.get('normalize', True)
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        logger.info("Analyzing topic frequency over time...")
        
        if 'cluster' not in df.columns:
            logger.warning("No cluster information available")
            return {}
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by time period
        if self.time_granularity == 'day':
            df['time_period'] = df['date'].dt.date
        elif self.time_granularity == 'week':
            df['time_period'] = df['date'].dt.to_period('W')
        else:  # month
            df['time_period'] = df['date'].dt.to_period('M')
        
        # Count articles per topic per source per time period
        frequency = df.groupby(['source', 'time_period', 'cluster']).size().reset_index(name='count')
        
        if self.normalize:
            # Normalize by total articles per source per period
            totals = df.groupby(['source', 'time_period']).size().reset_index(name='total')
            frequency = frequency.merge(totals, on=['source', 'time_period'])
            frequency['normalized_count'] = frequency['count'] / frequency['total']
        
        return {
            'frequency_data': frequency,
            'time_granularity': self.time_granularity
        }


class CoverageGapAnalyzer(AgendaAnalyzer):
    """Analyze coverage gaps between sources"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        cg_config = config.get('coverage_gap', {})
        self.gap_threshold = cg_config.get('gap_threshold', 0.3)
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        logger.info("Analyzing coverage gaps...")
        
        if 'cluster' not in df.columns:
            logger.warning("No cluster information available")
            return {}
        
        # Get topic coverage per source
        topic_coverage = df.groupby(['source', 'cluster']).size().unstack(fill_value=0)
        
        # Normalize
        topic_coverage_norm = topic_coverage.div(topic_coverage.sum(axis=1), axis=0)
        
        # Find gaps (topics covered significantly more by one source)
        sources = topic_coverage_norm.index.tolist()
        gaps = {}
        
        for topic in topic_coverage_norm.columns:
            coverage = topic_coverage_norm[topic]
            
            # Check for significant differences
            max_coverage = coverage.max()
            min_coverage = coverage.min()
            
            if max_coverage - min_coverage > self.gap_threshold:
                # Convert topic to string for JSON compatibility
                topic_key = str(topic)
                gaps[topic_key] = {
                    'max_source': coverage.idxmax(),
                    'max_coverage': float(max_coverage),
                    'min_source': coverage.idxmin(),
                    'min_coverage': float(min_coverage),
                    'gap': float(max_coverage - min_coverage)
                }
        
        return {
            'coverage_matrix': topic_coverage_norm.to_dict(),
            'significant_gaps': gaps,
            'num_gaps': len(gaps)
        }


class TopicProminenceAnalyzer(AgendaAnalyzer):
    """Analyze topic prominence based on various features"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        tp_config = config.get('topic_prominence', {})
        self.features = tp_config.get('features', ['headline_presence', 'article_length'])
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        logger.info("Analyzing topic prominence...")
        
        if 'cluster' not in df.columns:
            logger.warning("No cluster information available")
            return {}
        
        prominence_scores = {}
        
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            
            topic_prominence = {}
            
            for topic in source_df['cluster'].unique():
                topic_df = source_df[source_df['cluster'] == topic]
                
                scores = {}
                
                # Headline presence (topic keywords in headline)
                if 'headline_presence' in self.features:
                    if 'topic_keywords' in topic_df.columns:
                        # Simplified: just count
                        scores['headline_presence'] = len(topic_df) / len(source_df)
                
                # Article length
                if 'article_length' in self.features:
                    avg_length = topic_df['article'].str.split().str.len().mean()
                    scores['avg_article_length'] = avg_length
                
                # Early mention (topic appears in first paragraph)
                if 'early_mention' in self.features:
                    early_mentions = 0
                    for article in topic_df['article']:
                        first_para = article.split('\n')[0]
                        # Simplified check
                        early_mentions += 1 if len(first_para) > 100 else 0
                    scores['early_mention_rate'] = early_mentions / len(topic_df)
                
                # Combine scores
                prominence = np.mean(list(scores.values())) if scores else 0
                topic_prominence[topic] = {
                    'individual_scores': scores,
                    'combined_prominence': prominence
                }
            
            prominence_scores[source] = topic_prominence
        
        return {
            'prominence_by_source': prominence_scores
        }


class TemporalFocusAnalyzer(AgendaAnalyzer):
    """Analyze temporal focus and lag in coverage"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.tf_config = config.get('temporal_focus', {})
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        logger.info("Analyzing temporal focus...")
        
        if 'cluster' not in df.columns or 'date' not in df.columns:
            logger.warning("Missing cluster or date information")
            return {}
        
        df['date'] = pd.to_datetime(df['date'])
        
        # For each topic, find first mention per source
        topic_lags = {}
        
        for topic in df['cluster'].unique():
            topic_df = df[df['cluster'] == topic]
            
            first_mentions = topic_df.groupby('source')['date'].min()
            
            # Calculate lag from earliest mention
            earliest = first_mentions.min()
            lags = (first_mentions - earliest).dt.days
            
            topic_lags[topic] = {
                'first_mentions': first_mentions.to_dict(),
                'lags_days': lags.to_dict(),
                'earliest_source': first_mentions.idxmin(),
                'latest_source': first_mentions.idxmax()
            }
        
        return {
            'topic_lags': topic_lags
        }


class TopicPersistenceAnalyzer(AgendaAnalyzer):
    """Analyze how long topics remain in coverage"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        persist_config = config.get('topic_persistence', {})
        self.window_size = persist_config.get('window_size', 14)
        self.min_mentions = persist_config.get('min_mentions', 3)
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        logger.info("Analyzing topic persistence...")
        
        if 'cluster' not in df.columns or 'date' not in df.columns:
            logger.warning("Missing cluster or date information")
            return {}
        
        df['date'] = pd.to_datetime(df['date'])
        
        persistence_data = {}
        
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            
            topic_persistence = {}
            
            for topic in source_df['cluster'].unique():
                topic_df = source_df[source_df['cluster'] == topic].sort_values('date')
                
                if len(topic_df) < self.min_mentions:
                    continue
                
                # Calculate persistence (days between first and last mention)
                first_date = topic_df['date'].min()
                last_date = topic_df['date'].max()
                duration = (last_date - first_date).days
                
                # Count mentions over time
                mentions = len(topic_df)
                
                topic_persistence[topic] = {
                    'duration_days': duration,
                    'total_mentions': mentions,
                    'mention_rate': mentions / (duration + 1) if duration > 0 else mentions
                }
            
            persistence_data[source] = topic_persistence
        
        return {
            'persistence_by_source': persistence_data
        }


class AgendaAnalysisFactory:
    """Factory for creating agenda analyzers"""
    
    @staticmethod
    def create(config: Dict, analyses: List[str]) -> List[AgendaAnalyzer]:
        """
        Create agenda analyzers based on configuration
        
        Args:
            config: Configuration dictionary
            analyses: List of analysis names
            
        Returns:
            List of AgendaAnalyzer instances
        """
        analyzers_map = {
            'topic_frequency': TopicFrequencyAnalyzer,
            'coverage_gap': CoverageGapAnalyzer,
            'topic_prominence': TopicProminenceAnalyzer,
            'temporal_focus': TemporalFocusAnalyzer,
            'topic_persistence': TopicPersistenceAnalyzer
        }
        
        analyzers = []
        for analysis in analyses:
            if analysis in analyzers_map:
                analyzers.append(analyzers_map[analysis](config))
            else:
                logger.warning(f"Unknown agenda analysis: {analysis}")
        
        return analyzers
