# DiffOutlet: Media Bias Analysis Pipeline

A comprehensive, modular pipeline for analyzing media bias across different news outlets through multiple analytical lenses.

## Features

### 8 Plug-and-Play Components

1. **Event Detection & Clustering**

   - LDA, NMF, BERTopic
   - Semantic clustering with sentence embeddings
   - Temporal-semantic clustering
   - Headline similarity clustering

2. **Lexical Choice Extraction**

   - TF-IDF keyword extraction
   - RAKE and YAKE algorithms
   - POS-filtered term extraction
   - Contextual word detection

3. **Sentiment & Stance Analysis**

   - VADER, TextBlob
   - Aspect-based sentiment
   - Transformer-based analysis
   - Emotion classification
   - Sentiment trajectory

4. **Framing Analysis**

   - Generic frame detection (conflict, economic, morality, etc.)
   - Metaphor detection
   - Agency and voice analysis
   - Moral foundations analysis
   - Quote attribution

5. **Embedding-based Comparison**

   - Contextualized embeddings (BERT, RoBERTa)
   - Word embedding shifts
   - Semantic divergence measurement

6. **Agenda-Setting Analysis**

   - Topic frequency over time
   - Coverage gap detection
   - Topic prominence
   - Temporal focus and lag
   - Topic persistence

7. **Visualization**

   - Lexical difference heatmaps
   - Sentiment comparison charts
   - Topic-source network graphs
   - Time series dashboards
   - Contrasting word clouds

8. **Comparative Metrics**
   - Bias scores
   - Framing divergence indices
   - Lexical diversity measures
   - Objectivity scores

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone or download this repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required NLTK data:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Configure Your Pipeline

Edit `config.yaml` to enable/disable components and set their parameters:

```yaml
# Enable/disable components
event_detection:
  enabled: true
  method: "semantic_clustering" # Choose your method

lexical_extraction:
  enabled: true
  methods:
    - "tfidf_keywords"
    - "rake"

sentiment_analysis:
  enabled: true
  methods:
    - "vader"
    - "aspect_based"
# ... and so on
```

### 2. Prepare Your Data

Place your CSV files in the `Dataset` folder. The pipeline expects files with these columns:

- `headline`: Article headline
- `date`: Publication date
- `link`: Article URL
- `source`: News source name
- `categories`: Article category
- `description`: Full article text

Update dataset paths in `config.yaml`:

```yaml
data:
  dataset_paths:
    - "../Dataset/your_dataset1.csv"
    - "../Dataset/your_dataset2.csv"
```

### 3. Run the Pipeline

```bash
python main.py
```

## Pipeline Architecture

```
Data Loading → Text Preprocessing → Component Processing → Visualization → Results Export
                                            ↓
                        ┌───────────────────────────────────────┐
                        │  Plug-and-Play Components              │
                        ├───────────────────────────────────────┤
                        │  1. Event Detection & Clustering       │
                        │  2. Lexical Choice Extraction         │
                        │  3. Sentiment & Stance Analysis       │
                        │  4. Framing Analysis                  │
                        │  5. Embedding-based Comparison        │
                        │  6. Agenda-Setting Analysis           │
                        │  7. Visualization                     │
                        │  8. Comparative Metrics               │
                        └───────────────────────────────────────┘
```

## Configuration Options

### Plug-and-Play Components

Each component has multiple algorithm options. Example:

**Event Detection:**

- `lda` - Latent Dirichlet Allocation
- `nmf` - Non-negative Matrix Factorization
- `bertopic` - BERTopic modeling
- `semantic_clustering` - K-means on embeddings
- `temporal_semantic` - Time + semantic clustering
- `headline_similarity` - Clustering by headline similarity

**Sentiment Analysis:**

- `vader` - VADER sentiment analyzer
- `textblob` - TextBlob polarity/subjectivity
- `aspect_based` - Sentiment toward specific aspects
- `transformer` - Transformer-based models
- `emotion_classification` - Emotion detection
- `sentiment_trajectory` - Sentiment changes across article

### Processing Options

```yaml
processing:
  text_cleaning:
    remove_urls: true
    lowercase: true
    lemmatize: true

  parallelization:
    enabled: true
    n_jobs: -1 # Use all cores

  caching:
    enabled: true
```

## Output

Results are saved in the `results` directory:

1. **Processed Data**: CSV file with all extracted features
2. **Results JSON**: Complete analysis results
3. **Summary Report**: Human-readable summary
4. **Visualizations**: PNG/SVG/PDF files in `visualizations` directory

## Example Configuration Scenarios

### Scenario 1: Quick Analysis (Fast)

```yaml
event_detection:
  enabled: true
  method: "lda"

lexical_extraction:
  enabled: true
  methods: ["tfidf_keywords"]

sentiment_analysis:
  enabled: true
  methods: ["vader"]

# Disable heavy components
embedding_comparison:
  enabled: false
```

### Scenario 2: Deep Analysis (Comprehensive)

```yaml
event_detection:
  enabled: true
  method: "bertopic"

lexical_extraction:
  enabled: true
  methods: ["tfidf_keywords", "rake", "yake", "pos_filtered"]

sentiment_analysis:
  enabled: true
  methods: ["vader", "aspect_based", "transformer", "sentiment_trajectory"]

framing_analysis:
  enabled: true
  methods: ["generic_frames", "metaphor_detection", "moral_foundations"]

embedding_comparison:
  enabled: true
  methods: ["contextualized_embeddings", "word_embedding_shifts"]
# All components enabled
```

## Extending the Pipeline

### Adding a New Method

1. Create a new class in the appropriate component file
2. Inherit from the base class
3. Implement the required methods
4. Add to the Factory class

Example:

```python
class MyNewClusteringMethod(EventDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        # Your initialization

    def detect_events(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your implementation
        return df

# Add to EventDetectionFactory
detectors = {
    # ... existing methods
    'my_new_method': MyNewClusteringMethod
}
```

### Adding a New Component

1. Create a new file in `components/`
2. Implement analyzer classes with base class
3. Create a Factory class
4. Import in `main.py`
5. Add configuration section in `config.yaml`

## Troubleshooting

### Common Issues

**Import errors:**

```bash
pip install -r requirements.txt --upgrade
```

**NLTK data not found:**

```python
python -c "import nltk; nltk.download('all')"
```

**Memory issues with large datasets:**

- Reduce sample sizes in config
- Enable caching
- Process sources separately

**Slow processing:**

- Enable parallelization
- Use faster methods (e.g., VADER instead of transformers)
- Reduce max_features in vectorizers

## Performance Tips

1. **For large datasets (>100K articles):**

   - Use sampling in event detection
   - Enable caching
   - Use incremental processing

2. **For faster execution:**

   - Disable heavy transformers
   - Use TF-IDF instead of embeddings
   - Limit visualization outputs

3. **For better accuracy:**
   - Use transformer-based models
   - Enable multiple extraction methods
   - Increase topic numbers for clustering

## Citation

If you use this pipeline in your research, please cite:

```
@software{diffoutlet2024,
  title={DiffOutlet: A Modular Pipeline for Media Bias Analysis},
  year={2024},
  author={Your Name},
}
```

## License

This project is available for academic and research purposes.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

Built using:

- scikit-learn
- NLTK, spaCy
- Gensim, BERTopic
- Sentence Transformers
- Hugging Face Transformers
- And many other excellent libraries
