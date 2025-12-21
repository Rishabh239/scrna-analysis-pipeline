"""
scRNA-seq Analysis Pipeline

A comprehensive pipeline for single-cell RNA sequencing analysis including
preprocessing, clustering, differential expression, pathway enrichment,
and machine learning classification.

Modules:
    - data_loader: Load 10x Genomics data
    - preprocessing: QC, normalization, clustering
    - differential_expression: Marker gene analysis
    - enrichment: GO and KEGG pathway analysis
    - classification: ML cell type classification
    - cli: Command-line interface

Author: scRNA Analysis Pipeline
License: MIT
"""

__version__ = "1.0.0"

from .data_loader import load_10x_data, load_h5ad, save_h5ad
from .preprocessing import ScRNAPreprocessor, preprocess_adata
from .differential_expression import DifferentialExpression, find_markers
from .enrichment import EnrichmentAnalyzer, run_enrichment
from .classification import CellTypeClassifier

__all__ = [
    'load_10x_data',
    'load_h5ad',
    'save_h5ad',
    'ScRNAPreprocessor',
    'preprocess_adata',
    'DifferentialExpression',
    'find_markers',
    'EnrichmentAnalyzer',
    'run_enrichment',
    'CellTypeClassifier'
]
