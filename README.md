# 🧬 Single-Cell RNA-Seq Analysis Pipeline

A comprehensive Python pipeline for analyzing single-cell RNA sequencing (scRNA-seq) data, featuring preprocessing, clustering, differential expression analysis, pathway enrichment, and machine learning-based cell type classification.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scanpy](https://img.shields.io/badge/Scanpy-1.9%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-red)

## 📋 Overview

This pipeline processes 10x Genomics single-cell RNA-seq data through a complete analysis workflow:

1. **Data Loading** - Import 10x Genomics filtered feature-barcode matrices
2. **Quality Control** - Filter cells based on gene counts and mitochondrial content
3. **Normalization** - Normalize and scale gene expression data
4. **Dimensionality Reduction** - PCA, UMAP, and t-SNE visualization
5. **Clustering** - Leiden algorithm for cell population identification
6. **Differential Expression** - Identify marker genes for each cluster
7. **Pathway Analysis** - GO and KEGG enrichment analysis
8. **Cell Type Classification** - ML models (Random Forest, SVM, Neural Network)

## 🏗️ Project Structure

```
scrna-analysis-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load 10x Genomics data
│   ├── preprocessing.py     # QC, normalization, clustering
│   ├── differential_expression.py  # Marker gene analysis
│   ├── enrichment.py        # GO and KEGG pathway analysis
│   ├── classification.py    # ML cell type classification
│   ├── visualization.py     # Plotting functions
│   ├── utils.py             # Utility functions
│   └── cli.py               # Command-line interface
├── data/
│   └── filtered_feature_bc_matrix/  # Place your 10x data here
│       ├── barcodes.tsv.gz
│       ├── features.tsv.gz
│       └── matrix.mtx.gz
├── models/                  # Saved ML models
├── results/                 # Output plots and files
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── examples/                # Example notebooks
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scrna-analysis-pipeline.git
cd scrna-analysis-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Your Data

1. Download or obtain your 10x Genomics data (e.g., from [10x Genomics datasets](https://www.10xgenomics.com/resources/datasets))
2. Extract the `.tar.gz` file:
   ```bash
   tar -xvzf your_data.tar.gz -C data/
   ```
3. Ensure the following files are in `data/filtered_feature_bc_matrix/`:
   - `barcodes.tsv.gz`
   - `features.tsv.gz`
   - `matrix.mtx.gz`

### Run the Pipeline

```bash
# Run complete pipeline
python -m src.cli run --input data/filtered_feature_bc_matrix --output results/

# Run only preprocessing
python -m src.cli preprocess --input data/filtered_feature_bc_matrix --output results/

# Run only classification (requires preprocessed data)
python -m src.cli classify --input results/preprocessed.h5ad --output results/

# Train ML models
python -m src.cli train --input results/preprocessed.h5ad --model-dir models/
```

### Python API

```python
from src.data_loader import load_10x_data
from src.preprocessing import preprocess_adata, run_clustering
from src.classification import CellTypeClassifier

# Load data
adata = load_10x_data("data/filtered_feature_bc_matrix")

# Preprocess
adata = preprocess_adata(adata)

# Cluster and visualize
adata = run_clustering(adata)

# Train classifier
classifier = CellTypeClassifier()
classifier.train(adata)
predictions = classifier.predict(new_adata)
```

## 📊 Pipeline Stages

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
- Cross-cluster heatmaps for comparison

### 6. Machine Learning Classification

| Model | Description | Use Case |
|-------|-------------|----------|
| Random Forest | Ensemble tree-based | Fast, interpretable |
| SVM | Support Vector Machine with PCA | High accuracy |
| Neural Network | Deep learning with regularization | Complex patterns |

## 📁 Input Data Format

The pipeline expects 10x Genomics **filtered feature-barcode matrix** format:

```
filtered_feature_bc_matrix/
├── barcodes.tsv.gz    # Cell barcodes (one per line)
├── features.tsv.gz    # Gene IDs, names, and types
└── matrix.mtx.gz      # Sparse expression matrix (Market Matrix format)
```

## 📤 Output Files

### Preprocessed Data
- `preprocessed.h5ad` - Scanpy AnnData object with all analyses

### Visualizations (in `results/`)
- `umap_leiden.png` - UMAP colored by cluster
- `umap_cell_type.png` - UMAP colored by cell type
- `tsne_leiden.png` - t-SNE visualization
- `pca_variance_ratio.png` - PCA explained variance
- `rank_genes.png` - Top marker genes per cluster

### Enrichment Results
- `GO_enrichment_results_cluster_X.csv` - GO terms per cluster
- `KEGG_enrichment_results_cluster_X.csv` - KEGG pathways per cluster
- `go_heatmap.png` - GO enrichment heatmap
- `kegg_heatmap.png` - KEGG enrichment heatmap

### ML Models (in `models/`)
- `cell_classifier.h5` - Trained neural network
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Cell type label encoder
- `confusion_matrix.png` - Model performance
- `shap_summary.png` - Feature importance (SHAP)

## ⚙️ Configuration

Edit `config.yaml` to customize parameters:

```yaml
qc:
  min_genes: 200
  max_counts: 25000
  max_mt_pct: 5

clustering:
  n_neighbors: 10
  n_pcs: 40
  resolution: 0.5

classification:
  test_size: 0.2
  n_estimators: 100
  epochs: 50
```

## 🔬 Cell Type Annotation

Default cell type mapping (customize in config):

| Cluster | Cell Type |
|---------|-----------|
| 0 | T cells |
| 1 | B cells |
| 2 | Macrophages |
| 3 | Dendritic cells |
| 4 | NK cells |

Use marker genes to validate and adjust annotations for your dataset.

## 📚 Dependencies

- **scanpy** - Single-cell analysis
- **pandas**, **numpy** - Data manipulation
- **scikit-learn** - Machine learning
- **tensorflow/keras** - Deep learning
- **gseapy** - Enrichment analysis
- **matplotlib**, **seaborn** - Visualization
- **shap** - Model interpretability

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📖 References

- [Scanpy Documentation](https://scanpy.readthedocs.io/)
- [10x Genomics](https://www.10xgenomics.com/)
- [PBMC Dataset](https://www.10xgenomics.com/resources/datasets)

## ⚠️ Disclaimer

This tool is for research purposes only. Results should be validated by domain experts before biological interpretation or clinical application.
