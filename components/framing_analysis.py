"""
Component 4: Framing Analysis
Detects narrative frames and perspectives
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
import logging

logger = logging.getLogger(__name__)


class FramingAnalyzer:
    """Base class for framing analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze framing in articles"""
        raise NotImplementedError


class GenericFrameDetector(FramingAnalyzer):
    """Detect generic news frames"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        gf_config = config.get('generic_frames', {})
        self.frames = gf_config.get('frames', ['conflict', 'human_interest', 'economic', 'morality'])
        
        # Define frame lexicons
        self.frame_lexicons = {
            'conflict': ['conflict', 'disagree', 'dispute', 'fight', 'battle', 'clash', 'oppose', 'versus', 'against', 'war', 'attack', 'criticism', 'confrontation'],
            'human_interest': ['personal', 'human', 'story', 'individual', 'family', 'emotional', 'feelings', 'victim', 'survivor', 'people', 'life', 'experience'],
            'economic': ['economy', 'financial', 'economic', 'money', 'cost', 'price', 'budget', 'tax', 'business', 'market', 'profit', 'loss', 'trade', 'growth'],
            'morality': ['moral', 'ethics', 'right', 'wrong', 'justice', 'values', 'principles', 'should', 'ought', 'duty', 'responsibility', 'fair', 'unfair'],
            'responsibility': ['responsible', 'accountability', 'blame', 'fault', 'cause', 'solution', 'action', 'policy', 'decision', 'leadership']
        }
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Detecting generic frames: {self.frames}...")
        
        for frame in self.frames:
            if frame in self.frame_lexicons:
                lexicon = self.frame_lexicons[frame]
                
                # Count frame indicators
                df[f'frame_{frame}_count'] = df['article'].apply(
                    lambda text: sum(1 for word in lexicon if word in text.lower())
                )
                
                # Normalize by article length
                df[f'frame_{frame}_density'] = df[f'frame_{frame}_count'] / (df['article'].str.split().str.len() + 1)
        
        # Determine dominant frame
        frame_cols = [f'frame_{frame}_density' for frame in self.frames if frame in self.frame_lexicons]
        if frame_cols:
            df['dominant_frame'] = df[frame_cols].idxmax(axis=1).str.replace('frame_', '').str.replace('_density', '')
        
        return df


class MetaphorDetector(FramingAnalyzer):
    """Detect metaphorical language"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        metaphor_config = config.get('metaphor_detection', {})
        self.metaphor_types = metaphor_config.get('metaphor_types', ['war', 'disease', 'journey'])
        
        # Define metaphor patterns
        self.metaphor_patterns = {
            'war': ['battle', 'fight', 'combat', 'weapon', 'army', 'defeat', 'victory', 'enemy', 'attack', 'defend', 'strategy', 'campaign'],
            'disease': ['virus', 'infected', 'spread', 'epidemic', 'cure', 'heal', 'sick', 'disease', 'contagion', 'symptom', 'diagnosis'],
            'journey': ['path', 'road', 'journey', 'direction', 'forward', 'backward', 'progress', 'destination', 'navigate', 'route'],
            'building': ['foundation', 'build', 'construct', 'structure', 'framework', 'pillar', 'collapse', 'rebuild', 'infrastructure']
        }
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Detecting metaphors: {self.metaphor_types}...")
        
        for metaphor_type in self.metaphor_types:
            if metaphor_type in self.metaphor_patterns:
                patterns = self.metaphor_patterns[metaphor_type]
                
                df[f'metaphor_{metaphor_type}_count'] = df['article'].apply(
                    lambda text: sum(1 for pattern in patterns if pattern in text.lower())
                )
                
                df[f'metaphor_{metaphor_type}_present'] = df[f'metaphor_{metaphor_type}_count'] > 0
        
        return df


class AgencyAnalyzer(FramingAnalyzer):
    """Analyze agency and voice in text"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
        except:
            self.nlp = None
        
        self.agency_config = config.get('agency_analysis', {})
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.nlp:
            logger.warning("Spacy not available, skipping agency analysis")
            return df
        
        logger.info("Analyzing agency and voice...")
        
        active_counts = []
        passive_counts = []
        subject_types = []
        
        for text in df['article'][:100]:  # Limit for performance
            doc = self.nlp(text[:10000])
            
            active = 0
            passive = 0
            subjects = []
            
            for sent in doc.sents:
                # Detect passive voice
                if any(token.dep_ == 'nsubjpass' for token in sent):
                    passive += 1
                else:
                    active += 1
                
                # Extract subjects
                for token in sent:
                    if token.dep_ in ['nsubj', 'nsubjpass']:
                        subjects.append(token.text)
            
            active_counts.append(active)
            passive_counts.append(passive)
            subject_types.append(subjects[:5])  # Top 5 subjects
        
        df['active_voice_count'] = active_counts + [0] * (len(df) - len(active_counts))
        df['passive_voice_count'] = passive_counts + [0] * (len(df) - len(passive_counts))
        df['active_voice_ratio'] = df['active_voice_count'] / (df['active_voice_count'] + df['passive_voice_count'] + 1)
        
        return df


class MoralFoundationsAnalyzer(FramingAnalyzer):
    """Analyze moral foundations in text"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        mf_config = config.get('moral_foundations', {})
        self.foundations = mf_config.get('foundations', ['care_harm', 'fairness', 'loyalty', 'authority', 'sanctity'])
        
        # Simplified moral foundations lexicon
        self.mf_lexicon = {
            'care_harm': ['care', 'compassion', 'harm', 'hurt', 'suffer', 'protect', 'safe', 'danger', 'cruel', 'kind'],
            'fairness': ['fair', 'unfair', 'justice', 'rights', 'equality', 'bias', 'discriminate', 'equal', 'deserve'],
            'loyalty': ['loyal', 'betray', 'team', 'together', 'united', 'patriot', 'country', 'nation', 'duty'],
            'authority': ['authority', 'respect', 'tradition', 'order', 'obey', 'disobey', 'hierarchy', 'leader', 'rule'],
            'sanctity': ['sacred', 'pure', 'holy', 'sin', 'disgust', 'decent', 'moral', 'integrity', 'virtue', 'corrupt']
        }
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Analyzing moral foundations: {self.foundations}...")
        
        for foundation in self.foundations:
            if foundation in self.mf_lexicon:
                lexicon = self.mf_lexicon[foundation]
                
                df[f'moral_{foundation}_count'] = df['article'].apply(
                    lambda text: sum(1 for word in lexicon if word in text.lower())
                )
                
                df[f'moral_{foundation}_density'] = df[f'moral_{foundation}_count'] / (df['article'].str.split().str.len() + 1)
        
        return df


class QuoteAttributionAnalyzer(FramingAnalyzer):
    """Analyze quote attribution patterns"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.qa_config = config.get('quote_attribution', {})
        self.min_quote_length = self.qa_config.get('min_quote_length', 10)
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analyzing quote attribution...")
        
        quote_counts = []
        quote_lengths = []
        quoted_sources = []
        
        # Simple quote detection using regex
        quote_pattern = r'["\']([^"\']{' + str(self.min_quote_length) + r',})["\']'
        
        for text in df['article']:
            quotes = re.findall(quote_pattern, text)
            quote_counts.append(len(quotes))
            
            if quotes:
                avg_length = sum(len(q.split()) for q in quotes) / len(quotes)
                quote_lengths.append(avg_length)
            else:
                quote_lengths.append(0)
            
            # Extract attribution (simplified)
            sources = []
            for quote in quotes:
                # Look for text before quote
                pattern = r'(\w+\s+\w+)\s+(?:said|stated|claimed|argued|explained)[,:]?\s*["\']' + re.escape(quote[:20])
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    sources.append(match.group(1))
            
            quoted_sources.append(sources)
        
        df['quote_count'] = quote_counts
        df['avg_quote_length'] = quote_lengths
        df['quoted_sources'] = quoted_sources
        df['num_quoted_sources'] = [len(sources) for sources in quoted_sources]
        
        return df


class FramingAnalysisFactory:
    """Factory for creating framing analyzers"""
    
    @staticmethod
    def create(config: Dict, methods: List[str]) -> List[FramingAnalyzer]:
        """
        Create framing analyzers based on configuration
        
        Args:
            config: Configuration dictionary
            methods: List of method names
            
        Returns:
            List of FramingAnalyzer instances
        """
        analyzers_map = {
            'generic_frames': GenericFrameDetector,
            'metaphor_detection': MetaphorDetector,
            'agency_analysis': AgencyAnalyzer,
            'moral_foundations': MoralFoundationsAnalyzer,
            'quote_attribution': QuoteAttributionAnalyzer
        }
        
        analyzers = []
        for method in methods:
            if method in analyzers_map:
                analyzers.append(analyzers_map[method](config))
            else:
                logger.warning(f"Unknown framing analysis method: {method}")
        
        return analyzers


def compare_framing_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare framing patterns across sources
    
    Args:
        df: DataFrame with framing columns
        
    Returns:
        DataFrame with framing comparison per source
    """
    framing_cols = [col for col in df.columns if 'frame_' in col or 'metaphor_' in col or 'moral_' in col]
    
    numeric_cols = [col for col in framing_cols if df[col].dtype in [np.float64, np.int64]]
    
    if numeric_cols:
        comparison = df.groupby('source')[numeric_cols].mean()
        return comparison
    
    return pd.DataFrame()
