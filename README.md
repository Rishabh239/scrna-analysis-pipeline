#  Single-Cell RNA-Seq Analysis Pipeline

A comprehensive Python pipeline for analyzing single-cell RNA sequencing (scRNA-seq) data, featuring preprocessing, clustering, differential expression analysis, pathway enrichment, and machine learning-based cell type classification.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Scanpy](https://img.shields.io/badge/Scanpy-1.9%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-red)

##  Overview

This pipeline processes 10x Genomics single-cell RNA-seq data through a complete analysis workflow:

1. **Data Loading** - Import 10x Genomics filtered feature-barcode matrices
2. **Quality Control** - Filter cells based on gene counts and mitochondrial content
3. **Normalization** - Normalize and scale gene expression data
4. **Dimensionality Reduction** - PCA, UMAP, and t-SNE visualization
5. **Clustering** - Leiden algorithm for cell population identification
6. **Differential Expression** - Identify marker genes for each cluster
7. **Pathway Analysis** - GO and KEGG enrichment analysis with heatmaps
8. **Cell Type Classification** - ML models (Random Forest, SVM, Neural Network)

##  Project Structure

```
scrna-analysis-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load 10x Genomics data
│   ├── preprocessing.py        # QC, normalization, clustering
│   ├── differential_expression.py   # Marker gene analysis
│   ├── enrichment.py           # GO and KEGG pathway analysis
│   ├── classification.py       # ML cell type classification
│   └── cli.py                  # Command-line interface
├── data/
│   └── filtered_feature_bc_matrix/  # Place your 10x data here
│       ├── barcodes.tsv.gz
│       ├── features.tsv.gz
│       └── matrix.mtx.gz
├── models/                     # Saved ML models
├── results/                    # Output plots and files
├── tests/                      # Unit tests
├── docs/                       # Documentation
└── examples/                   # Example notebooks
```

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rishabh239/scrna-analysis-pipeline.git
cd scrna-analysis-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Your Data

1. Download your 10x Genomics data (e.g., from [10x Genomics datasets](https://www.10xgenomics.com/resources/datasets))
2. Extract the `.tar.gz` file:
   ```bash
   # Linux/Mac
   tar -xvzf your_data.tar.gz -C data/
   
   # Windows: Use 7-Zip to extract to data/ folder
   ```
3. Ensure the following files are in `data/filtered_feature_bc_matrix/`:
   - `barcodes.tsv.gz`
   - `features.tsv.gz`
   - `matrix.mtx.gz`

### Run the Pipeline

```bash
# Run complete pipeline (preprocessing + enrichment + classification)
python -m src.cli run --input data/filtered_feature_bc_matrix --output results/ --annotate --classify

# Run only preprocessing
python -m src.cli preprocess --input data/filtered_feature_bc_matrix --output results/

# Run only classification (requires preprocessed data)
python -m src.cli classify --input results/preprocessed.h5ad --output results/

# Predict cell types with trained model
python -m src.cli predict --input new_data.h5ad --model-dir models/ --output predictions.csv
```

### Python API

```python
from src.data_loader import load_10x_data
from src.preprocessing import ScRNAPreprocessor
from src.differential_expression import DifferentialExpression
from src.enrichment import EnrichmentAnalyzer
from src.classification import CellTypeClassifier

# Load data
adata = load_10x_data("data/filtered_feature_bc_matrix")

# Preprocess
preprocessor = ScRNAPreprocessor(results_dir="results/")
adata = preprocessor.run_full_pipeline(adata)

# Find marker genes
de = DifferentialExpression(results_dir="results/")
adata = de.find_marker_genes(adata)

# Run enrichment analysis
enrichment = EnrichmentAnalyzer(results_dir="results/enrichment/")
cluster_results = enrichment.run_enrichment_all_clusters(adata, n_genes=100)
enrichment.create_enrichment_heatmap(cluster_results, 'go')
enrichment.create_enrichment_heatmap(cluster_results, 'kegg')

# Train classifier
classifier = CellTypeClassifier()
classifier.train(adata, model_type='all')
classifier.save_model('cell_classifier')

# Predict on new data
predictions = classifier.predict(new_adata)
```

##  Pipeline Stages

### 1. Quality Control

Filters cells based on:
- Minimum genes per cell (default: 200)
- Maximum total counts (default: 25,000)
- Maximum mitochondrial gene percentage (default: 5%)

### 2. Normalization & Scaling

- Normalizes counts to 10,000 per cell
- Log-transforms expression values
- Identifies highly variable genes
- Scales to unit variance

### 3. Dimensionality Reduction & Clustering

- **PCA**: 50 principal components
- **UMAP/t-SNE**: 2D visualization
- **Leiden clustering**: Identifies cell populations

### 4. Differential Expression

- Wilcoxon rank-sum test for marker genes
- Handles NaN/Inf values in log fold changes
- Identifies top markers per cluster

### 5. Pathway Enrichment

- **GO Biological Process**: Gene Ontology analysis
- **KEGG Pathways**: Metabolic and signaling pathways
- **Heatmaps**: Cross-cluster enrichment comparison
- **Bar Charts**: Top enriched terms per cluster

### 6. Machine Learning Classification

| Model | Description | Use Case |
|-------|-------------|----------|
| Random Forest | Ensemble tree-based | Fast, interpretable |
| SVM | Support Vector Machine with PCA | High accuracy |
| Neural Network | Deep learning with regularization | Complex patterns |

##  Input Data Format

The pipeline expects 10x Genomics **filtered feature-barcode matrix** format:

```
filtered_feature_bc_matrix/
├── barcodes.tsv.gz    # Cell barcodes (one per line)
├── features.tsv.gz    # Gene IDs, names, and types
└── matrix.mtx.gz      # Sparse expression matrix (Market Matrix format)
```

##  Output Files

After running the full pipeline, you'll get:

```
results/
├── preprocessing/
│   ├── qc_metrics_before.png       # QC violin plots before filtering
│   ├── qc_metrics_after.png        # QC violin plots after filtering
│   ├── pca_variance_ratio.png      # PCA explained variance
│   ├── umap_leiden.png             # UMAP colored by cluster
│   ├── umap_cell_type.png          # UMAP colored by cell type
│   └── tsne_leiden.png             # t-SNE visualization
│
├── differential_expression/
│   ├── marker_genes.csv            # Top marker genes per cluster
│   └── rank_genes_marker_genes.png # Marker gene visualization
│
├── enrichment/
│   ├── GO_enrichment_cluster_0.csv     # GO results for cluster 0
│   ├── GO_enrichment_cluster_1.csv     # GO results for cluster 1
│   ├── KEGG_enrichment_cluster_0.csv   # KEGG results for cluster 0
│   ├── KEGG_enrichment_cluster_1.csv   # KEGG results for cluster 1
│   ├── go_heatmap.png                  # GO enrichment heatmap (all clusters)
│   ├── kegg_heatmap.png                # KEGG enrichment heatmap (all clusters)
│   ├── go_enrichment_cluster_0.png     # GO bar chart for cluster 0
│   ├── go_enrichment_cluster_1.png     # GO bar chart for cluster 1
│   ├── kegg_enrichment_cluster_0.png   # KEGG bar chart for cluster 0
│   └── kegg_enrichment_cluster_1.png   # KEGG bar chart for cluster 1
│
├── classification/
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_neural_network.png
│   ├── training_history.png            # Neural network training curves
│   └── shap_summary_random_forest.png  # Feature importance (SHAP)
│
├── models/
│   ├── cell_classifier_random_forest.pkl
│   ├── cell_classifier_svm.pkl
│   ├── cell_classifier_neural_network.h5
│   ├── cell_classifier_scaler.pkl
│   ├── cell_classifier_label_encoder.pkl
│   └── cell_classifier_pca.pkl
│
└── processed_data.h5ad                 # Complete processed dataset
```

##  Configuration

Edit `config.yaml` to customize parameters:

```yaml
# Quality Control
qc:
  min_genes: 200
  max_counts: 25000
  max_mt_pct: 5

# Clustering
clustering:
  n_neighbors: 10
  n_pcs: 40
  resolution: 0.5

# Machine Learning
ml:
  test_size: 0.2
  random_state: 42
  use_smote: true
  
  random_forest:
    n_estimators: 100
  
  neural_network:
    epochs: 50
    batch_size: 32

# Enrichment Analysis
enrichment:
  n_genes: 100
  organism: 'human'
```

##  Cell Type Annotation

Default cell type mapping (customize based on your marker genes):

| Cluster | Cell Type |
|---------|-----------|
| 0 | T cells |
| 1 | B cells |
| 2 | Macrophages |
| 3 | Dendritic cells |
| 4 | NK cells |
| 5 | Monocytes |

**Note**: Clusters without mappings are automatically labeled as `Unknown_Cluster_X`. Update the mapping in `config.yaml` or `src/cli.py` based on your marker gene analysis.

##  Dependencies

Core dependencies:
- **scanpy** >= 1.9.0 - Single-cell analysis
- **pandas** >= 1.5.0 - Data manipulation
- **numpy** >= 1.23.0 - Numerical computing
- **scikit-learn** >= 1.1.0 - Machine learning
- **tensorflow** >= 2.10.0 - Deep learning
- **gseapy** >= 1.0.0 - Enrichment analysis
- **matplotlib** >= 3.6.0 - Visualization
- **seaborn** >= 0.12.0 - Statistical visualization
- **shap** >= 0.41.0 - Model interpretability
- **imbalanced-learn** >= 0.10.0 - SMOTE for class balancing

Install all dependencies:
```bash
pip install -r requirements.txt
```

##  Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

##  References

- [Scanpy Documentation](https://scanpy.readthedocs.io/)
- [10x Genomics](https://www.10xgenomics.com/)
- [PBMC Dataset](https://www.10xgenomics.com/resources/datasets)
- [GSEApy Documentation](https://gseapy.readthedocs.io/)

##  Disclaimer

This tool is for research purposes only. Results should be validated by domain experts before biological interpretation or clinical application.

---

**Author**: Rishabh  
**Institution**: Northeastern University  
**Field**: Bioinformatics / Computational Biology
