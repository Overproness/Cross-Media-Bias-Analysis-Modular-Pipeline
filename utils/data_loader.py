"""
Data Loader Module
Loads and preprocesses news articles from multiple sources
"""

import pandas as pd
import yaml
from typing import List, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial processing of news datasets"""
    
    def __init__(self, config: Dict):
        """
        Initialize DataLoader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.columns_map = self.data_config['columns']
        
    def load_datasets(self) -> pd.DataFrame:
        """
        Load all datasets specified in configuration
        
        Returns:
            Combined DataFrame with all articles
        """
        logger.info("Loading datasets...")
        
        all_data = []
        
        for dataset_path in self.data_config['dataset_paths']:
            try:
                # Resolve relative paths
                path = Path(__file__).parent.parent / dataset_path
                
                logger.info(f"Loading {path.name}...")
                
                # Try multiple encodings to handle different file formats
                df = None
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        # Use dtype_backend='pyarrow' for memory efficiency with string data
                        df = pd.read_csv(path, low_memory=False, encoding=encoding, dtype_backend='pyarrow')
                        if encoding != 'utf-8':
                            logger.info(f"Loaded {path.name} using {encoding} encoding")
                        break
                    except (UnicodeDecodeError, Exception):
                        try:
                            # Fallback to standard loading if pyarrow fails
                            df = pd.read_csv(path, low_memory=False, encoding=encoding)
                            if encoding != 'utf-8':
                                logger.info(f"Loaded {path.name} using {encoding} encoding (standard mode)")
                            break
                        except UnicodeDecodeError:
                            continue
                
                if df is None:
                    logger.error(f"Could not decode {path.name} with any encoding")
                    continue
                
                # Standardize column names
                df = self._standardize_columns(df)
                
                # Extract source from filename if not present
                if 'source' not in df.columns or df['source'].isna().all():
                    source_name = self._extract_source_name(path.name)
                    df['source'] = source_name
                
                all_data.append(df)
                logger.info(f"Loaded {len(df)} articles from {path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {dataset_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No datasets could be loaded")
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total articles loaded: {len(combined_df)}")
        
        return combined_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names according to configuration
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        rename_map = {}
        
        for std_name, actual_name in self.columns_map.items():
            if actual_name in df.columns:
                rename_map[actual_name] = std_name
        
        df = df.rename(columns=rename_map)
        
        # Ensure required columns exist
        required = ['headline', 'article', 'date', 'source']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing columns: {missing}")
        
        return df
    
    def _extract_source_name(self, filename: str) -> str:
        """
        Extract source name from filename
        
        Args:
            filename: Name of the file
            
        Returns:
            Source name
        """
        # Remove common prefixes and extensions
        name = filename.replace('Copy of ', '').replace('.csv', '')
        
        # Extract main source name
        if 'business_recorder' in name.lower():
            return 'Business Recorder'
        elif 'daily_times' in name.lower():
            return 'Daily Times'
        elif 'dawn' in name.lower():
            return 'Dawn'
        elif 'pakistan_today' in name.lower():
            return 'Pakistan Today'
        else:
            return name.replace('_', ' ').title()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        original_len = len(df)
        
        # Remove duplicates (in-place to save memory)
        df.drop_duplicates(subset=['headline', 'article'], keep='first', inplace=True)
        logger.info(f"Removed {original_len - len(df)} duplicates")
        
        # Handle missing values (in-place)
        df.dropna(subset=['headline', 'article'], inplace=True)
        logger.info(f"Removed {original_len - len(df)} rows with missing headline/article")
        
        # Reset index after removals
        df.reset_index(drop=True, inplace=True)
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        # Ensure article and headline columns are string type using PyArrow backend to avoid memory explosion
        df['article'] = df['article'].astype('string[pyarrow]')
        df['headline'] = df['headline'].astype('string[pyarrow]')
        
        # Filter by article length (memory efficient approach)
        min_length = self.data_config.get('min_article_length', 100)
        logger.info(f"Calculating article word counts...")
        word_counts = df['article'].str.split().str.len()
        
        # Create boolean mask for filtering
        mask = word_counts >= min_length
        logger.info(f"Filtering {mask.sum()} articles with >= {min_length} words (removing {(~mask).sum()})...")
        
        # Filter in-place to avoid memory allocation
        df = df[mask]
        df['article_word_count'] = word_counts[mask].values  # Use .values to avoid alignment issues
        df.reset_index(drop=True, inplace=True)  # Reset index after filtering
        
        logger.info(f"Filtered to {len(df)} articles with >= {min_length} words")
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # Clean text fields
        df['headline'] = df['headline'].str.strip()
        df['article'] = df['article'].str.strip()
        
        logger.info(f"Final dataset size: {len(df)} articles")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Sources: {df['source'].value_counts().to_dict()}")
        
        return df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_articles': len(df),
            'sources': df['source'].nunique(),
            'source_distribution': df['source'].value_counts().to_dict(),
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            },
            'avg_article_length': df['article_word_count'].mean() if 'article_word_count' in df.columns else None,
            'categories': df['category'].value_counts().head(10).to_dict() if 'category' in df.columns else {}
        }
        
        return stats


def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(__file__).parent.parent / config_path
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
