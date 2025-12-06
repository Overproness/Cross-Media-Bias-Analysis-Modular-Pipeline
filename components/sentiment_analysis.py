"""
Component 3: Sentiment & Stance Analysis
Measures emotional tone toward actors/topics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Base class for sentiment analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment in articles"""
        raise NotImplementedError


class VADERSentimentAnalyzer(SentimentAnalyzer):
    """Sentiment analysis using VADER"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.vader_config = config.get('vader', {})
            self.threshold = self.vader_config.get('compound_threshold', 0.05)
        except ImportError:
            logger.error("vaderSentiment not installed")
            self.analyzer = None
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.analyzer:
            raise ImportError("vaderSentiment not available")
        
        logger.info("Analyzing sentiment using VADER...")
        
        sentiments = []
        for text in df['article']:
            scores = self.analyzer.polarity_scores(text)
            sentiments.append(scores)
        
        # Add sentiment columns
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        df['sentiment_positive'] = [s['pos'] for s in sentiments]
        df['sentiment_negative'] = [s['neg'] for s in sentiments]
        df['sentiment_neutral'] = [s['neu'] for s in sentiments]
        
        # Classify sentiment
        df['sentiment_label'] = df['sentiment_compound'].apply(
            lambda x: 'positive' if x >= self.threshold 
            else 'negative' if x <= -self.threshold 
            else 'neutral'
        )
        
        return df


class TextBlobSentimentAnalyzer(SentimentAnalyzer):
    """Sentiment analysis using TextBlob"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
            tb_config = config.get('textblob', {})
            self.threshold = tb_config.get('polarity_threshold', 0.1)
        except ImportError:
            logger.warning("textblob not installed, skipping TextBlob sentiment analysis")
            self.TextBlob = None
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.TextBlob:
            logger.info("Skipping TextBlob analysis (textblob not installed)")
            return df
        
        logger.info("Analyzing sentiment using TextBlob...")
        
        polarities = []
        subjectivities = []
        
        for text in df['article']:
            blob = self.TextBlob(str(text))
            polarities.append(blob.sentiment.polarity)
            subjectivities.append(blob.sentiment.subjectivity)
        
        df['sentiment_polarity'] = polarities
        df['sentiment_subjectivity'] = subjectivities
        
        df['sentiment_label'] = df['sentiment_polarity'].apply(
            lambda x: 'positive' if x >= self.threshold 
            else 'negative' if x <= -self.threshold 
            else 'neutral'
        )
        
        return df


class AspectBasedSentimentAnalyzer(SentimentAnalyzer):
    """Aspect-based sentiment analysis"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        aspect_config = config.get('aspect_based', {})
        self.aspects = aspect_config.get('aspects', ['government', 'economy', 'policy'])
        self.base_model = aspect_config.get('model', 'vader')
        
        # Initialize base sentiment analyzer
        if self.base_model == 'vader':
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.base_model == 'textblob':
            from textblob import TextBlob
            self.TextBlob = TextBlob
            self.analyzer = None
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Analyzing aspect-based sentiment for aspects: {self.aspects}...")
        
        import re
        
        aspect_sentiments = {aspect: [] for aspect in self.aspects}
        total = len(df)
        
        for idx, text in enumerate(df['article']):
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{total} articles ({idx/total*100:.1f}%)")
            
            text_lower = text.lower()
            
            for aspect in self.aspects:
                # Find sentences containing the aspect
                sentences = [s for s in text.split('.') if aspect.lower() in s.lower()]
                
                if sentences:
                    aspect_text = ' '.join(sentences)
                    
                    if self.base_model == 'vader' and self.analyzer:
                        score = self.analyzer.polarity_scores(aspect_text)['compound']
                    elif self.base_model == 'textblob' and self.TextBlob:
                        score = self.TextBlob(aspect_text).sentiment.polarity
                    else:
                        score = 0
                else:
                    score = None  # Aspect not mentioned
                
                aspect_sentiments[aspect].append(score)
        
        # Add aspect sentiment columns
        for aspect in self.aspects:
            df[f'sentiment_{aspect}'] = aspect_sentiments[aspect]
        
        return df


class TransformerSentimentAnalyzer(SentimentAnalyzer):
    """Sentiment analysis using transformer models"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        try:
            from transformers import pipeline
            import torch
            
            transformer_config = config.get('transformer', {})
            model_name = transformer_config.get('model_name', 'cardiffnlp/twitter-roberta-base-sentiment')
            
            # Auto-detect GPU
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info(f"Using GPU for transformer sentiment: {torch.cuda.get_device_name(0)}")
            
            # Create pipeline with batch processing enabled
            self.pipeline = pipeline('sentiment-analysis', model=model_name, device=device, batch_size=32)
        except ImportError:
            logger.error("transformers not installed")
            self.pipeline = None
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.pipeline:
            raise ImportError("transformers not available")
        
        logger.info("Analyzing sentiment using transformer model...")
        
        # Truncate texts to 512 tokens (transformer limit)
        texts = [text[:512] for text in df['article'].tolist()]
        total = len(texts)
        
        logger.info(f"Processing {total} articles with batch processing...")
        
        # Use the pipeline's built-in batching (configured in __init__)
        sentiments = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            if i % (batch_size * 10) == 0 and i > 0:
                logger.info(f"Processed {i}/{total} articles ({i/total*100:.1f}%)")
            
            batch = texts[i:i+batch_size]
            results = self.pipeline(batch)
            sentiments.extend(results)
        
        df['sentiment_label'] = [s['label'].lower() for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        
        return df


class EmotionClassifier(SentimentAnalyzer):
    """Emotion classification"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        emotion_config = config.get('emotion_classification', {})
        self.emotions = emotion_config.get('emotions', ['anger', 'fear', 'joy', 'sadness'])
        self.model_type = emotion_config.get('model', 'nrc')
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Classifying emotions: {self.emotions}...")
        
        if self.model_type == 'nrc':
            # Simple lexicon-based approach (placeholder)
            emotion_lexicons = {
                'anger': ['angry', 'furious', 'outrage', 'rage'],
                'fear': ['afraid', 'fear', 'scared', 'terror'],
                'joy': ['happy', 'joy', 'delighted', 'pleased'],
                'sadness': ['sad', 'depressed', 'sorrow', 'grief'],
                'surprise': ['surprised', 'shocked', 'amazed'],
                'trust': ['trust', 'faith', 'confident']
            }
            
            for emotion in self.emotions:
                if emotion in emotion_lexicons:
                    lexicon = emotion_lexicons[emotion]
                    df[f'emotion_{emotion}'] = df['article'].apply(
                        lambda text: sum(1 for word in lexicon if word in text.lower())
                    )
        
        return df


class SentimentTrajectoryAnalyzer(SentimentAnalyzer):
    """Analyze sentiment trajectory across article"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        
        traj_config = config.get('sentiment_trajectory', {})
        self.segments = traj_config.get('segments', ['intro', 'middle', 'conclusion'])
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analyzing sentiment trajectory...")
        
        for segment in self.segments:
            sentiments = []
            
            for text in df['article']:
                sentences = text.split('.')
                n_sentences = len(sentences)
                
                if segment == 'intro':
                    segment_text = '.'.join(sentences[:max(1, n_sentences//3)])
                elif segment == 'middle':
                    start = n_sentences//3
                    end = 2*n_sentences//3
                    segment_text = '.'.join(sentences[start:end])
                else:  # conclusion
                    segment_text = '.'.join(sentences[2*n_sentences//3:])
                
                score = self.analyzer.polarity_scores(segment_text)['compound']
                sentiments.append(score)
            
            df[f'sentiment_{segment}'] = sentiments
        
        # Calculate trajectory change
        if 'sentiment_intro' in df.columns and 'sentiment_conclusion' in df.columns:
            df['sentiment_trajectory_change'] = df['sentiment_conclusion'] - df['sentiment_intro']
        
        return df


class StanceDetectionAnalyzer(SentimentAnalyzer):
    """Stance detection toward specific targets"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        stance_config = config.get('stance_detection', {})
        self.targets = stance_config.get('targets', ['government', 'opposition', 'policy'])
        self.model_type = stance_config.get('model', 'rule_based')
        
        # Initialize sentiment analyzer for rule-based approach
        if self.model_type == 'rule_based':
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                logger.error("vaderSentiment not installed")
                self.analyzer = None
        else:
            self.analyzer = None
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Detecting stance toward targets: {self.targets}...")
        
        stance_results = {target: [] for target in self.targets}
        
        for text in df['article']:
            text_lower = text.lower()
            
            for target in self.targets:
                # Find sentences mentioning the target
                sentences = [s for s in text.split('.') if target.lower() in s.lower()]
                
                if sentences and self.analyzer:
                    # Analyze sentiment of target-related sentences
                    target_text = ' '.join(sentences)
                    score = self.analyzer.polarity_scores(target_text)['compound']
                    
                    # Classify stance
                    if score >= 0.1:
                        stance = 'favor'
                    elif score <= -0.1:
                        stance = 'against'
                    else:
                        stance = 'neutral'
                else:
                    stance = 'not_mentioned'
                    score = None
                
                stance_results[target].append({'stance': stance, 'score': score})
        
        # Add stance columns
        for target in self.targets:
            df[f'stance_{target}'] = [r['stance'] for r in stance_results[target]]
            df[f'stance_{target}_score'] = [r['score'] for r in stance_results[target]]
        
        return df


class SubjectivityAnalyzer(SentimentAnalyzer):
    """Analyze subjectivity/objectivity of text"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        subj_config = config.get('subjectivity', {})
        self.method = subj_config.get('method', 'textblob')
        
        if self.method == 'textblob':
            try:
                from textblob import TextBlob
                self.TextBlob = TextBlob
            except ImportError:
                logger.warning("textblob not installed, subjectivity analysis will be skipped")
                self.TextBlob = None
        else:
            self.TextBlob = None
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Analyzing text subjectivity...")
        
        if not self.TextBlob:
            logger.info("Skipping subjectivity analysis (textblob not installed)")
            return df
        
        subjectivity_scores = []
        objectivity_labels = []
        
        for text in df['article']:
            blob = self.TextBlob(str(text))
            subj_score = blob.sentiment.subjectivity
            subjectivity_scores.append(subj_score)
            
            # Classify as objective/subjective
            if subj_score < 0.3:
                objectivity_labels.append('objective')
            elif subj_score > 0.6:
                objectivity_labels.append('subjective')
            else:
                objectivity_labels.append('mixed')
        
        df['subjectivity_score'] = subjectivity_scores
        df['objectivity_label'] = objectivity_labels
        
        return df


class SentimentAnalysisFactory:
    """Factory for creating sentiment analyzers"""
    
    @staticmethod
    def create(config: Dict, methods: List[str]) -> List[SentimentAnalyzer]:
        """
        Create sentiment analyzers based on configuration
        
        Args:
            config: Configuration dictionary
            methods: List of method names
            
        Returns:
            List of SentimentAnalyzer instances
        """
        analyzers_map = {
            'vader': VADERSentimentAnalyzer,
            'textblob': TextBlobSentimentAnalyzer,
            'aspect_based': AspectBasedSentimentAnalyzer,
            'transformer': TransformerSentimentAnalyzer,
            'emotion_classification': EmotionClassifier,
            'sentiment_trajectory': SentimentTrajectoryAnalyzer,
            'stance_detection': StanceDetectionAnalyzer,
            'subjectivity': SubjectivityAnalyzer
        }
        
        analyzers = []
        for method in methods:
            if method in analyzers_map:
                analyzers.append(analyzers_map[method](config))
            else:
                logger.warning(f"Unknown sentiment analysis method: {method}")
        
        return analyzers


def aggregate_sentiment_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment statistics by source
    
    Args:
        df: DataFrame with sentiment columns
        
    Returns:
        DataFrame with aggregated sentiment per source
    """
    sentiment_cols = [col for col in df.columns if 'sentiment' in col]
    
    agg_dict = {}
    for col in sentiment_cols:
        if df[col].dtype in [np.float64, np.int64]:
            agg_dict[col] = ['mean', 'std']
    
    if 'sentiment_label' in df.columns:
        agg_dict['sentiment_label'] = lambda x: x.value_counts().to_dict()
    
    aggregated = df.groupby('source').agg(agg_dict)
    
    return aggregated
