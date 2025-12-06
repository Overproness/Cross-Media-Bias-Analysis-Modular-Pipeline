"""
Text Preprocessing Module
Handles text cleaning and preprocessing
"""

import re
import string
from typing import List, Dict
import logging

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError:
    nltk = None

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text cleaning and preprocessing operations"""
    
    def __init__(self, config: Dict):
        """
        Initialize TextPreprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('processing', {}).get('text_cleaning', {})
        
        # Initialize NLP tools
        self.lemmatizer = None
        self.nlp = None
        self.stop_words = set()
        
        if nltk:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                logger.warning("NLTK data not found. Some features may not work.")
        
        if spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("Spacy model not found. Run: python -m spacy download en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text according to configuration
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        if self.config.get('remove_html', True):
            text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        if self.config.get('remove_urls', True):
            text = re.sub(r'http\S+|www.\S+', ' ', text)
        
        # Remove special characters
        if self.config.get('remove_special_chars', True):
            # Keep basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if nltk and word_tokenize:
            return word_tokenize(text)
        else:
            # Fallback to simple split
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not self.config.get('remove_stopwords', True):
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if not self.config.get('lemmatize', True) or not self.lemmatizer:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, return_string: bool = True):
        """
        Full preprocessing pipeline
        
        Args:
            text: Input text
            return_string: If True, return joined string; else return token list
            
        Returns:
            Preprocessed text or tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        if return_string:
            return ' '.join(tokens)
        return tokens
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if nltk and sent_tokenize:
            return sent_tokenize(text)
        else:
            # Fallback to simple split
            return re.split(r'[.!?]+', text)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of entities with type and text
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_pos_tags(self, text: str, allowed_pos: List[str] = None) -> List[Dict]:
        """
        Extract POS tags from text
        
        Args:
            text: Input text
            allowed_pos: List of allowed POS tags (e.g., ['NOUN', 'VERB'])
            
        Returns:
            List of tokens with POS tags
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            if allowed_pos and token.pos_ not in allowed_pos:
                continue
            
            tokens.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_
            })
        
        return tokens


def download_nltk_resources():
    """Download required NLTK resources"""
    if not nltk:
        logger.warning("NLTK not installed - some text preprocessing features will be limited")
        return
    
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download {resource}: {e}")
