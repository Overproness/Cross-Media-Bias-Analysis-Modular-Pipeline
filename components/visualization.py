"""
Component 7: Visualization & Interpretation
Makes patterns human-readable
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """Base class for visualization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = Path(self.viz_config.get('output_dir', 'visualizations'))
        self.output_dir.mkdir(exist_ok=True)
        self.save_format = self.viz_config.get('save_format', 'png')
        self.dpi = self.viz_config.get('dpi', 300)
    
    def visualize(self, data, **kwargs):
        """Create visualization"""
        raise NotImplementedError


class LexicalHeatmapVisualizer(Visualizer):
    """Create heatmap of lexical differences"""
    
    def visualize(self, lexical_data: Dict, output_name: str = 'lexical_heatmap'):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib/seaborn not installed")
            return
        
        logger.info("Creating lexical difference heatmap...")
        
        top_n = self.viz_config.get('lexical_heatmap', {}).get('top_n_terms', 50)
        
        # Prepare data matrix
        sources = list(lexical_data.keys())
        all_keywords = set()
        
        for source_data in lexical_data.values():
            all_keywords.update(source_data['top_20'][:top_n])
        
        # Create matrix
        matrix_data = []
        for keyword in list(all_keywords)[:top_n]:
            row = []
            for source in sources:
                keywords_dict = lexical_data[source]['keywords']
                row.append(keywords_dict.get(keyword, 0))
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data, columns=sources, index=list(all_keywords)[:top_n])
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df, cmap='RdBu_r', center=0, annot=False, cbar_kws={'label': 'TF-IDF Score'})
        plt.title('Lexical Choices Across Sources')
        plt.xlabel('Source')
        plt.ylabel('Keywords')
        plt.tight_layout()
        
        output_path = self.output_dir / f'{output_name}.{self.save_format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heatmap to {output_path}")


class SentimentComparisonVisualizer(Visualizer):
    """Create sentiment comparison charts"""
    
    def visualize(self, df: pd.DataFrame, output_name: str = 'sentiment_comparison'):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib/seaborn not installed")
            return
        
        logger.info("Creating sentiment comparison chart...")
        
        chart_type = self.viz_config.get('sentiment_comparison', {}).get('chart_type', 'bar')
        
        # Aggregate sentiment by source
        if 'sentiment_label' in df.columns:
            sentiment_dist = df.groupby(['source', 'sentiment_label']).size().unstack(fill_value=0)
            
            # Normalize
            sentiment_dist = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0) * 100
            
            if chart_type == 'bar':
                sentiment_dist.plot(kind='bar', figsize=(10, 6), colormap='RdYlGn')
                plt.title('Sentiment Distribution by Source')
                plt.xlabel('Source')
                plt.ylabel('Percentage (%)')
                plt.legend(title='Sentiment')
                plt.tight_layout()
            
            elif chart_type == 'radar':
                # Radar chart (simplified for 3 categories)
                from math import pi
                
                categories = list(sentiment_dist.columns)
                N = len(categories)
                
                angles = [n / float(N) * 2 * pi for n in range(N)]
                angles += angles[:1]
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                
                for source in sentiment_dist.index:
                    values = sentiment_dist.loc[source].values.tolist()
                    values += values[:1]
                    ax.plot(angles, values, 'o-', linewidth=2, label=source)
                    ax.fill(angles, values, alpha=0.25)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 100)
                plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                plt.title('Sentiment Comparison (Radar Chart)')
            
            output_path = self.output_dir / f'{output_name}.{self.save_format}'
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved sentiment chart to {output_path}")


class TopicNetworkVisualizer(Visualizer):
    """Create network graph of topics and sources"""
    
    def visualize(self, df: pd.DataFrame, output_name: str = 'topic_network'):
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.error("matplotlib/networkx not installed")
            return
        
        logger.info("Creating topic network graph...")
        
        if 'cluster' not in df.columns:
            logger.warning("No cluster information available")
            return
        
        # Create network
        G = nx.Graph()
        
        # Add nodes for sources and topics
        sources = df['source'].unique()
        topics = df['cluster'].unique()[:20]  # Limit topics
        
        for source in sources:
            G.add_node(source, node_type='source')
        
        for topic in topics:
            G.add_node(f'Topic_{topic}', node_type='topic')
        
        # Add edges (source-topic connections)
        min_edge_weight = self.viz_config.get('topic_network', {}).get('min_edge_weight', 0.3)
        
        for source in sources:
            source_topics = df[df['source'] == source]['cluster'].value_counts()
            total = source_topics.sum()
            
            for topic, count in source_topics.items():
                weight = count / total
                if weight >= min_edge_weight and topic in topics:
                    G.add_edge(source, f'Topic_{topic}', weight=weight)
        
        # Layout
        layout_type = self.viz_config.get('topic_network', {}).get('layout', 'spring')
        
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        else:  # kamada_kawai
            pos = nx.kamada_kawai_layout(G)
        
        # Draw
        plt.figure(figsize=(14, 10))
        
        # Separate node types
        source_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'source']
        topic_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'topic']
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='lightblue', 
                               node_size=1000, label='Sources')
        nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color='lightcoral',
                               node_size=500, label='Topics')
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Source-Topic Network')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        
        output_path = self.output_dir / f'{output_name}.{self.save_format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved network graph to {output_path}")


class TimeSeriesVisualizer(Visualizer):
    """Create time series dashboards"""
    
    def visualize(self, frequency_data: pd.DataFrame, output_name: str = 'time_series'):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        logger.info("Creating time series visualization...")
        
        # Plot topic coverage over time per source
        fig, axes = plt.subplots(len(frequency_data['source'].unique()), 1, 
                                figsize=(14, 4*len(frequency_data['source'].unique())))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for idx, source in enumerate(frequency_data['source'].unique()):
            source_data = frequency_data[frequency_data['source'] == source]
            
            # Pivot to get topics as columns
            pivot = source_data.pivot_table(
                index='time_period', 
                columns='cluster', 
                values='normalized_count' if 'normalized_count' in source_data.columns else 'count',
                fill_value=0
            )
            
            # Plot top topics
            top_topics = pivot.sum().nlargest(5).index
            pivot[top_topics].plot(ax=axes[idx], marker='o')
            
            axes[idx].set_title(f'{source} - Topic Coverage Over Time')
            axes[idx].set_xlabel('Time Period')
            axes[idx].set_ylabel('Coverage')
            axes[idx].legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / f'{output_name}.{self.save_format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved time series to {output_path}")


class WordCloudVisualizer(Visualizer):
    """Create contrasting word clouds"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        # Check dependencies during initialization
        self.wordcloud_available = False
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            self.wordcloud_available = True
        except ImportError:
            logger.warning("wordcloud package not installed. Install with: pip install wordcloud")
    
    def visualize(self, lexical_data: Dict, output_name: str = 'wordclouds'):
        if not self.wordcloud_available:
            logger.info("Skipping word cloud generation (wordcloud not installed)")
            return
        
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("wordcloud/matplotlib not installed")
            return
        
        logger.info("Creating word clouds...")
        
        sources = list(lexical_data.keys())
        n_sources = len(sources)
        
        if n_sources == 0:
            logger.warning("No lexical data available for word clouds")
            return
        
        logger.info(f"Creating word clouds for {n_sources} sources: {sources}")
        
        fig, axes = plt.subplots(1, n_sources, figsize=(6*n_sources, 6))
        
        if n_sources == 1:
            axes = [axes]
        
        for idx, source in enumerate(sources):
            keywords = lexical_data[source].get('keywords', {})
            
            if not keywords:
                logger.warning(f"No keywords found for source: {source}")
                continue
            
            logger.info(f"Generating word cloud for {source} with {len(keywords)} keywords")
            
            # Convert keywords to proper frequency format if needed
            # Ensure all values are numeric
            keyword_freq = {}
            for word, score in keywords.items():
                try:
                    keyword_freq[str(word)] = float(score)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid keyword: {word}={score}")
            
            if not keyword_freq:
                logger.warning(f"No valid keywords for {source} after conversion")
                continue
            
            wc = WordCloud(
                width=800, 
                height=400,
                background_color=self.viz_config.get('wordcloud_contrast', {}).get('background_color', 'white'),
                max_words=self.viz_config.get('wordcloud_contrast', {}).get('max_words', 100),
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(keyword_freq)
            
            axes[idx].imshow(wc, interpolation='bilinear')
            axes[idx].set_title(f'{source}', fontsize=16)
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f'{output_name}.{self.save_format}'
        logger.info(f"Saving word clouds to {output_path}")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Successfully saved word clouds to {output_path}")


class VisualizationFactory:
    """Factory for creating visualizers"""
    
    @staticmethod
    def create(config: Dict, outputs: List[str]) -> List[Visualizer]:
        """
        Create visualizers based on configuration
        
        Args:
            config: Configuration dictionary
            outputs: List of output types
            
        Returns:
            List of Visualizer instances
        """
        visualizers_map = {
            'lexical_heatmap': LexicalHeatmapVisualizer,
            'sentiment_comparison': SentimentComparisonVisualizer,
            'topic_network': TopicNetworkVisualizer,
            'time_series': TimeSeriesVisualizer,
            'wordcloud_contrast': WordCloudVisualizer
        }
        
        visualizers = []
        for output in outputs:
            if output in visualizers_map:
                visualizers.append(visualizers_map[output](config))
            else:
                logger.warning(f"Unknown visualization output: {output}")
        
        return visualizers
