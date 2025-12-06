"""
Main Pipeline Orchestrator
Coordinates all components of the media bias analysis pipeline
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import DataLoader, load_config
from utils.text_preprocessing import TextPreprocessor, download_nltk_resources
from components.event_detection import EventDetectionFactory
from components.lexical_extraction import LexicalExtractionFactory, compare_lexical_choices
from components.sentiment_analysis import SentimentAnalysisFactory, aggregate_sentiment_by_source
from components.framing_analysis import FramingAnalysisFactory, compare_framing_by_source
from components.embedding_comparison import EmbeddingComparisonFactory
from components.agenda_setting import AgendaAnalysisFactory
from components.visualization import VisualizationFactory
from components.comparative_metrics import ComparativeMetricsFactory


# Setup logging
def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.get('processing', {}).get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('log_file', 'pipeline.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class DiffOutletPipeline:
    """Main pipeline for media bias analysis"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        self.logger.info("=" * 80)
        self.logger.info("DiffOutlet Pipeline Initialized")
        self.logger.info("=" * 80)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.text_preprocessor = TextPreprocessor(self.config)
        
        # Results storage
        self.results = {}
        self.df = None
        
        # Create output directory
        output_dir = Path(self.config.get('output', {}).get('results_dir', 'results'))
        output_dir.mkdir(exist_ok=True)
        self.output_dir = output_dir
    
    def run(self):
        """Run the complete pipeline"""
        self.logger.info("Starting pipeline execution...")
        
        # Step 1: Load and preprocess data
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 1: Data Loading and Preprocessing")
        self.logger.info("=" * 80)
        self._load_data()
        
        # Step 2: Event detection and clustering
        if self.config.get('event_detection', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 2: Event Detection & Clustering")
            self.logger.info("=" * 80)
            self._detect_events()
        
        # Step 3: Lexical choice extraction
        if self.config.get('lexical_extraction', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 3: Lexical Choice Extraction")
            self.logger.info("=" * 80)
            self._extract_lexical_choices()
        
        # Step 4: Sentiment analysis
        if self.config.get('sentiment_analysis', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 4: Sentiment & Stance Analysis")
            self.logger.info("=" * 80)
            self._analyze_sentiment()
        
        # Step 5: Framing analysis
        if self.config.get('framing_analysis', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 5: Framing Analysis")
            self.logger.info("=" * 80)
            self._analyze_framing()
        
        # Step 6: Embedding-based comparison
        if self.config.get('embedding_comparison', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 6: Embedding-based Comparison")
            self.logger.info("=" * 80)
            self._compare_embeddings()
        
        # Step 7: Agenda-setting analysis
        if self.config.get('agenda_setting', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 7: Agenda-Setting Analysis")
            self.logger.info("=" * 80)
            self._analyze_agenda()
        
        # Step 8: Comparative metrics
        if self.config.get('comparative_metrics', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 8: Comparative Metrics")
            self.logger.info("=" * 80)
            self._calculate_metrics()
        
        # Step 9: Visualization
        if self.config.get('visualization', {}).get('enabled', True):
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 9: Visualization")
            self.logger.info("=" * 80)
            self._create_visualizations()
        
        # Step 10: Save results
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 10: Saving Results")
        self.logger.info("=" * 80)
        self._save_results()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Pipeline execution completed successfully!")
        self.logger.info("=" * 80)
    
    def _load_data(self):
        """Load and preprocess data"""
        # Load datasets
        self.df = self.data_loader.load_datasets()
        
        # Preprocess
        self.df = self.data_loader.preprocess_data(self.df)
        
        # Get statistics
        stats = self.data_loader.get_dataset_statistics(self.df)
        self.results['dataset_stats'] = stats
        
        self.logger.info(f"Loaded {stats['total_articles']} articles from {stats['sources']} sources")
    
    def _detect_events(self):
        """Detect events and cluster articles"""
        event_config = self.config['event_detection']
        
        try:
            detector = EventDetectionFactory.create(event_config)
            self.df = detector.detect_events(self.df)
            
            n_clusters = self.df['cluster'].nunique()
            self.logger.info(f"Detected {n_clusters} event clusters")
            
            self.results['event_detection'] = {
                'n_clusters': n_clusters,
                'method': event_config.get('method')
            }
        except Exception as e:
            self.logger.error(f"Event detection failed: {e}")
    
    def _extract_lexical_choices(self):
        """Extract lexical choices"""
        lexical_config = self.config['lexical_extraction']
        methods = lexical_config.get('methods', [])
        
        try:
            extractors = LexicalExtractionFactory.create(lexical_config, methods)
            
            lexical_results = {}
            for extractor in extractors:
                method_name = extractor.__class__.__name__
                self.logger.info(f"Running {method_name}...")
                
                try:
                    result = extractor.extract(self.df)
                    lexical_results[method_name] = result
                except Exception as e:
                    self.logger.error(f"{method_name} failed: {e}")
            
            self.results['lexical_extraction'] = lexical_results
            
            # Compare lexical choices
            if lexical_results:
                first_result = list(lexical_results.values())[0]
                comparison = compare_lexical_choices(first_result)
                self.results['lexical_comparison'] = comparison
                
        except Exception as e:
            self.logger.error(f"Lexical extraction failed: {e}")
    
    def _analyze_sentiment(self):
        """Analyze sentiment"""
        sentiment_config = self.config['sentiment_analysis']
        methods = sentiment_config.get('methods', [])
        
        try:
            analyzers = SentimentAnalysisFactory.create(sentiment_config, methods)
            
            for analyzer in analyzers:
                method_name = analyzer.__class__.__name__
                self.logger.info(f"Running {method_name}...")
                
                try:
                    self.df = analyzer.analyze(self.df)
                except Exception as e:
                    self.logger.error(f"{method_name} failed: {e}")
            
            # Aggregate by source
            aggregated = aggregate_sentiment_by_source(self.df)
            self.results['sentiment_aggregated'] = aggregated
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
    
    def _analyze_framing(self):
        """Analyze framing"""
        framing_config = self.config['framing_analysis']
        methods = framing_config.get('methods', [])
        
        try:
            analyzers = FramingAnalysisFactory.create(framing_config, methods)
            
            for analyzer in analyzers:
                method_name = analyzer.__class__.__name__
                self.logger.info(f"Running {method_name}...")
                
                try:
                    self.df = analyzer.analyze(self.df)
                except Exception as e:
                    self.logger.error(f"{method_name} failed: {e}")
            
            # Compare framing by source
            comparison = compare_framing_by_source(self.df)
            self.results['framing_comparison'] = comparison
            
        except Exception as e:
            self.logger.error(f"Framing analysis failed: {e}")
    
    def _compare_embeddings(self):
        """Compare embeddings"""
        embedding_config = self.config['embedding_comparison']
        methods = embedding_config.get('methods', [])
        
        try:
            comparators = EmbeddingComparisonFactory.create(embedding_config, methods)
            
            embedding_results = {}
            for comparator in comparators:
                method_name = comparator.__class__.__name__
                self.logger.info(f"Running {method_name}...")
                
                try:
                    result = comparator.compare(self.df)
                    embedding_results[method_name] = result
                except Exception as e:
                    self.logger.error(f"{method_name} failed: {e}")
            
            self.results['embedding_comparison'] = embedding_results
            
        except Exception as e:
            self.logger.error(f"Embedding comparison failed: {e}")
    
    def _analyze_agenda(self):
        """Analyze agenda-setting"""
        agenda_config = self.config['agenda_setting']
        analyses = agenda_config.get('analyses', [])
        
        try:
            analyzers = AgendaAnalysisFactory.create(agenda_config, analyses)
            
            agenda_results = {}
            for analyzer in analyzers:
                method_name = analyzer.__class__.__name__
                self.logger.info(f"Running {method_name}...")
                
                try:
                    result = analyzer.analyze(self.df)
                    agenda_results[method_name] = result
                except Exception as e:
                    self.logger.error(f"{method_name} failed: {e}")
            
            self.results['agenda_setting'] = agenda_results
            
        except Exception as e:
            self.logger.error(f"Agenda-setting analysis failed: {e}")
    
    def _calculate_metrics(self):
        """Calculate comparative metrics"""
        metrics_config = self.config['comparative_metrics']
        metrics = metrics_config.get('metrics', [])
        
        try:
            calculators = ComparativeMetricsFactory.create(metrics_config, metrics)
            
            metrics_results = {}
            for calculator in calculators:
                method_name = calculator.__class__.__name__
                self.logger.info(f"Running {method_name}...")
                
                try:
                    # Pass additional data if available
                    kwargs = {}
                    # Only BiasScoreCalculator needs lexical_data
                    if method_name == 'BiasScoreCalculator' and 'lexical_extraction' in self.results:
                        first_result = list(self.results['lexical_extraction'].values())[0]
                        kwargs['lexical_data'] = first_result
                    
                    result = calculator.calculate(self.df, **kwargs)
                    metrics_results[method_name] = result
                except Exception as e:
                    self.logger.error(f"{method_name} failed: {e}")
            
            self.results['comparative_metrics'] = metrics_results
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
    
    def _create_visualizations(self):
        """Create visualizations"""
        viz_config = self.config['visualization']
        outputs = viz_config.get('outputs', [])
        
        try:
            visualizers = VisualizationFactory.create(self.config, outputs)
            
            for visualizer in visualizers:
                viz_name = visualizer.__class__.__name__
                self.logger.info(f"Creating {viz_name}...")
                
                try:
                    if 'Lexical' in viz_name and 'lexical_extraction' in self.results:
                        first_result = list(self.results['lexical_extraction'].values())[0]
                        visualizer.visualize(first_result)
                    
                    elif 'Sentiment' in viz_name:
                        visualizer.visualize(self.df)
                    
                    elif 'Network' in viz_name:
                        visualizer.visualize(self.df)
                    
                    elif 'TimeSeries' in viz_name and 'agenda_setting' in self.results:
                        for analyzer_result in self.results['agenda_setting'].values():
                            if 'frequency_data' in analyzer_result:
                                visualizer.visualize(analyzer_result['frequency_data'])
                                break
                    
                    elif 'WordCloud' in viz_name and 'lexical_extraction' in self.results:
                        first_result = list(self.results['lexical_extraction'].values())[0]
                        visualizer.visualize(first_result)
                    
                except Exception as e:
                    self.logger.error(f"{viz_name} failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
    
    def _save_results(self):
        """Save all results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processed DataFrame
        if self.config.get('output', {}).get('save_intermediate', True):
            csv_path = self.output_dir / f'processed_data_{timestamp}.csv'
            
            # Select columns to save (exclude complex objects)
            save_cols = [col for col in self.df.columns 
                        if not col.startswith('embedding') and 
                        not col.endswith('_keywords') and
                        self.df[col].dtype != object or col in ['headline', 'article', 'source', 'date']]
            
            self.df[save_cols].to_csv(csv_path, index=False)
            self.logger.info(f"Saved processed data to {csv_path}")
        
        # Save results dictionary
        export_formats = self.config.get('output', {}).get('export_formats', ['json'])
        
        if 'json' in export_formats:
            # Convert non-serializable objects
            json_results = self._prepare_for_json(self.results)
            json_path = self.output_dir / f'results_{timestamp}.json'
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Saved results to {json_path}")
        
        # Save summary report
        self._save_summary_report(timestamp)
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            # Convert tuple and numpy keys to strings for JSON compatibility
            new_dict = {}
            for k, v in obj.items():
                # Convert key to string if it's a tuple or numpy type
                if isinstance(k, tuple):
                    new_key = str(k)
                elif isinstance(k, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                    new_key = int(k)
                elif isinstance(k, (np.floating, np.float16, np.float32, np.float64)):
                    new_key = float(k)
                else:
                    new_key = k
                
                # Ensure key is JSON-compatible (str, int, float, bool, or None)
                if not isinstance(new_key, (str, int, float, bool, type(None))):
                    new_key = str(new_key)
                
                new_dict[new_key] = self._prepare_for_json(v)
            return new_dict
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Period, pd.Timestamp)):
            return str(obj)
        elif isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dict, then recursively prepare for JSON
            df_dict = obj.to_dict('list')
            return self._prepare_for_json(df_dict)
        elif isinstance(obj, pd.Series):
            return self._prepare_for_json(obj.tolist())
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)
    
    def _save_summary_report(self, timestamp):
        """Save a summary report"""
        report_path = self.output_dir / f'summary_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DiffOutlet Pipeline Summary Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset statistics
            if 'dataset_stats' in self.results:
                f.write("Dataset Statistics:\n")
                f.write("-" * 40 + "\n")
                stats = self.results['dataset_stats']
                f.write(f"Total Articles: {stats['total_articles']}\n")
                f.write(f"Number of Sources: {stats['sources']}\n")
                f.write(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}\n")
                f.write(f"Source Distribution:\n")
                for source, count in stats['source_distribution'].items():
                    f.write(f"  - {source}: {count}\n")
                f.write("\n")
            
            # Event detection
            if 'event_detection' in self.results:
                f.write("Event Detection:\n")
                f.write("-" * 40 + "\n")
                ed = self.results['event_detection']
                f.write(f"Method: {ed.get('method')}\n")
                f.write(f"Number of Clusters: {ed.get('n_clusters')}\n\n")
            
            # Comparative metrics
            if 'comparative_metrics' in self.results:
                f.write("Comparative Metrics:\n")
                f.write("-" * 40 + "\n")
                
                for metric_name, metric_data in self.results['comparative_metrics'].items():
                    f.write(f"\n{metric_name}:\n")
                    f.write(str(metric_data)[:500] + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Saved summary report to {report_path}")


def main():
    """Main entry point"""
    # Download NLTK resources
    download_nltk_resources()
    
    # Create and run pipeline
    pipeline = DiffOutletPipeline('config.yaml')
    pipeline.run()


if __name__ == '__main__':
    main()
