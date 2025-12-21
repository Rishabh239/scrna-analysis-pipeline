"""
Preprocessing Module for scRNA-seq Analysis Pipeline

This module handles quality control, normalization, dimensionality reduction,
and clustering of single-cell RNA-seq data.

Author: scRNA Analysis Pipeline
License: MIT
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScRNAPreprocessor:
    """
    Preprocessor class for single-cell RNA-seq data.
    
    This class provides methods for:
    - Quality control filtering
    - Normalization and scaling
    - Highly variable gene selection
    - Dimensionality reduction (PCA, UMAP, t-SNE)
    - Clustering (Leiden algorithm)
    - Cell type annotation
    
    Example:
        >>> preprocessor = ScRNAPreprocessor()
        >>> adata = preprocessor.run_full_pipeline(adata)
    """
    
    # Default QC parameters
    DEFAULT_QC_PARAMS = {
        'min_genes': 200,
        'max_genes': None,
        'min_counts': None,
        'max_counts': 25000,
        'max_mt_pct': 5.0,
        'min_cells': 3
    }
    
    # Default clustering parameters
    DEFAULT_CLUSTER_PARAMS = {
        'n_neighbors': 10,
        'n_pcs': 40,
        'resolution': 0.5
    }
    
    def __init__(self, 
                 qc_params: Optional[Dict] = None,
                 cluster_params: Optional[Dict] = None,
                 results_dir: str = 'results'):
        """
        Initialize the preprocessor.
        
        Args:
            qc_params: Quality control parameters
            cluster_params: Clustering parameters
            results_dir: Directory for saving plots
        """
        self.qc_params = {**self.DEFAULT_QC_PARAMS, **(qc_params or {})}
        self.cluster_params = {**self.DEFAULT_CLUSTER_PARAMS, **(cluster_params or {})}
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set scanpy figure directory
        sc.settings.figdir = str(self.results_dir)
        sc.settings.set_figure_params(dpi=100, facecolor='white')
    
    def run_full_pipeline(self, 
                          adata: sc.AnnData,
                          cell_type_dict: Optional[Dict[str, str]] = None) -> sc.AnnData:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            adata: Input AnnData object
            cell_type_dict: Optional mapping of cluster IDs to cell types
            
        Returns:
            Preprocessed AnnData object
        """
        logger.info("Starting full preprocessing pipeline")
        
        # Store raw counts
        adata.layers['counts'] = adata.X.copy()
        
        # Quality control
        adata = self.run_qc(adata)
        
        # Normalization
        adata = self.normalize(adata)
        
        # Find highly variable genes
        adata = self.find_variable_genes(adata)
        
        # Scale data
        adata = self.scale(adata)
        
        # Dimensionality reduction
        adata = self.run_pca(adata)
        adata = self.run_umap(adata)
        adata = self.run_tsne(adata)
        
        # Clustering
        adata = self.run_clustering(adata)
        
        # Annotate cell types
        if cell_type_dict:
            adata = self.annotate_cell_types(adata, cell_type_dict)
        
        logger.info("Preprocessing pipeline complete")
        return adata
    
    def run_qc(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Run quality control filtering.
        
        Filters cells based on:
        - Number of genes detected
        - Total counts
        - Mitochondrial gene percentage
        
        Args:
            adata: Input AnnData object
            
        Returns:
            Filtered AnnData object
        """
        logger.info("Running quality control...")
        initial_cells = adata.n_obs
        initial_genes = adata.n_vars
        
        # Calculate QC metrics
        # Identify mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            adata, 
            qc_vars=['mt'], 
            percent_top=None, 
            log1p=False, 
            inplace=True
        )
        
        # Plot QC metrics before filtering
        self._plot_qc_metrics(adata, suffix='_before')
        
        # Filter cells
        if self.qc_params['min_genes']:
            adata = adata[adata.obs.n_genes_by_counts >= self.qc_params['min_genes'], :]
        
        if self.qc_params['max_genes']:
            adata = adata[adata.obs.n_genes_by_counts <= self.qc_params['max_genes'], :]
        
        if self.qc_params['min_counts']:
            adata = adata[adata.obs.total_counts >= self.qc_params['min_counts'], :]
        
        if self.qc_params['max_counts']:
            adata = adata[adata.obs.total_counts <= self.qc_params['max_counts'], :]
        
        if self.qc_params['max_mt_pct']:
            adata = adata[adata.obs.pct_counts_mt <= self.qc_params['max_mt_pct'], :]
        
        # Filter genes
        if self.qc_params['min_cells']:
            sc.pp.filter_genes(adata, min_cells=self.qc_params['min_cells'])
        
        # Plot QC metrics after filtering
        self._plot_qc_metrics(adata, suffix='_after')
        
        logger.info(f"  Cells: {initial_cells:,} → {adata.n_obs:,} "
                   f"({initial_cells - adata.n_obs:,} removed)")
        logger.info(f"  Genes: {initial_genes:,} → {adata.n_vars:,} "
                   f"({initial_genes - adata.n_vars:,} removed)")
        
        return adata.copy()
    
    def _plot_qc_metrics(self, adata: sc.AnnData, suffix: str = '') -> None:
        """Plot QC metrics violin plots."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            sc.pl.violin(adata, 'n_genes_by_counts', ax=axes[0], show=False)
            axes[0].set_title('Genes per Cell')
            
            sc.pl.violin(adata, 'total_counts', ax=axes[1], show=False)
            axes[1].set_title('Total Counts')
            
            sc.pl.violin(adata, 'pct_counts_mt', ax=axes[2], show=False)
            axes[2].set_title('% Mitochondrial')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'qc_metrics{suffix}.png', dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot QC metrics: {e}")
    
    def normalize(self, adata: sc.AnnData, 
                  target_sum: float = 1e4) -> sc.AnnData:
        """
        Normalize gene expression data.
        
        Args:
            adata: Input AnnData object
            target_sum: Target sum for normalization
            
        Returns:
            Normalized AnnData object
        """
        logger.info("Normalizing data...")
        
        # Normalize counts per cell
        sc.pp.normalize_total(adata, target_sum=target_sum)
        
        # Log transform
        sc.pp.log1p(adata)
        
        logger.info(f"  Normalized to {target_sum:.0e} counts per cell")
        
        return adata
    
    def find_variable_genes(self, 
                            adata: sc.AnnData,
                            min_mean: float = 0.0125,
                            max_mean: float = 3,
                            min_disp: float = 0.5) -> sc.AnnData:
        """
        Identify highly variable genes.
        
        Args:
            adata: Input AnnData object
            min_mean: Minimum mean expression
            max_mean: Maximum mean expression
            min_disp: Minimum dispersion
            
        Returns:
            AnnData with highly_variable annotation
        """
        logger.info("Finding highly variable genes...")
        
        sc.pp.highly_variable_genes(
            adata,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp
        )
        
        n_hvg = sum(adata.var.highly_variable)
        logger.info(f"  Found {n_hvg:,} highly variable genes")
        
        # Plot
        try:
            sc.pl.highly_variable_genes(adata, save='_hvg.png', show=False)
        except:
            pass
        
        return adata
    
    def scale(self, adata: sc.AnnData, 
              max_value: float = 10) -> sc.AnnData:
        """
        Scale gene expression to unit variance.
        
        Args:
            adata: Input AnnData object
            max_value: Maximum value after scaling
            
        Returns:
            Scaled AnnData object
        """
        logger.info("Scaling data...")
        
        # Store normalized data before scaling
        adata.raw = adata
        
        # Scale
        sc.pp.scale(adata, max_value=max_value)
        
        logger.info(f"  Scaled with max_value={max_value}")
        
        return adata
    
    def run_pca(self, adata: sc.AnnData, 
                n_comps: int = 50) -> sc.AnnData:
        """
        Perform PCA dimensionality reduction.
        
        Args:
            adata: Input AnnData object
            n_comps: Number of principal components
            
        Returns:
            AnnData with PCA results
        """
        logger.info("Running PCA...")
        
        sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')
        
        # Plot variance ratio
        try:
            sc.pl.pca_variance_ratio(adata, log=True, save='_variance.png', show=False)
        except:
            pass
        
        logger.info(f"  Computed {n_comps} principal components")
        
        return adata
    
    def run_umap(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Compute UMAP embedding.
        
        Args:
            adata: Input AnnData object
            
        Returns:
            AnnData with UMAP coordinates
        """
        logger.info("Computing UMAP...")
        
        # Build neighbor graph
        sc.pp.neighbors(
            adata,
            n_neighbors=self.cluster_params['n_neighbors'],
            n_pcs=self.cluster_params['n_pcs']
        )
        
        # Compute UMAP
        sc.tl.umap(adata)
        
        logger.info("  UMAP complete")
        
        return adata
    
    def run_tsne(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Compute t-SNE embedding.
        
        Args:
            adata: Input AnnData object
            
        Returns:
            AnnData with t-SNE coordinates
        """
        logger.info("Computing t-SNE...")
        
        sc.tl.tsne(adata, n_pcs=self.cluster_params['n_pcs'])
        
        logger.info("  t-SNE complete")
        
        return adata
    
    def run_clustering(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Perform Leiden clustering.
        
        Args:
            adata: Input AnnData object
            
        Returns:
            AnnData with cluster assignments
        """
        logger.info("Running Leiden clustering...")
        
        sc.tl.leiden(adata, resolution=self.cluster_params['resolution'])
        
        n_clusters = len(adata.obs['leiden'].unique())
        logger.info(f"  Found {n_clusters} clusters")
        
        # Plot UMAP with clusters
        try:
            sc.pl.umap(adata, color='leiden', save='_leiden.png', show=False)
        except:
            pass
        
        return adata
    
    def annotate_cell_types(self, 
                            adata: sc.AnnData,
                            cell_type_dict: Dict[str, str]) -> sc.AnnData:
        """
        Annotate clusters with cell type labels.
        
        Args:
            adata: Input AnnData object
            cell_type_dict: Mapping of cluster IDs to cell type names
            
        Returns:
            AnnData with cell_type annotation
        """
        logger.info("Annotating cell types...")
        
        # Get all unique clusters
        clusters = adata.obs['leiden'].unique().tolist()
        
        # Create complete mapping with 'Unknown' for unmapped clusters
        complete_mapping = {}
        for cluster in clusters:
            if str(cluster) in cell_type_dict:
                complete_mapping[cluster] = cell_type_dict[str(cluster)]
            else:
                complete_mapping[cluster] = f'Unknown_Cluster_{cluster}'
                logger.warning(f"  Cluster {cluster} not in mapping, labeled as 'Unknown_Cluster_{cluster}'")
        
        # Map cluster IDs to cell types
        adata.obs['cell_type'] = adata.obs['leiden'].map(complete_mapping)
        
        # Convert to categorical
        adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
        
        # Log distribution
        for ct, count in adata.obs['cell_type'].value_counts().items():
            logger.info(f"  {ct}: {count:,} cells")
        
        # Plot UMAP with cell types
        try:
            sc.pl.umap(adata, color='cell_type', save='_cell_type.png', show=False)
        except:
            pass
        
        return adata
    
    def save_results(self, adata: sc.AnnData, 
                     filename: str = 'preprocessed.h5ad') -> None:
        """
        Save preprocessed data to file.
        
        Args:
            adata: AnnData object to save
            filename: Output filename
        """
        output_path = self.results_dir / filename
        adata.write_h5ad(output_path)
        logger.info(f"Saved preprocessed data to: {output_path}")


def preprocess_adata(adata: sc.AnnData, **kwargs) -> sc.AnnData:
    """
    Convenience function for preprocessing.
    
    Args:
        adata: Input AnnData object
        **kwargs: Additional arguments for ScRNAPreprocessor
        
    Returns:
        Preprocessed AnnData object
    """
    preprocessor = ScRNAPreprocessor(**kwargs)
    return preprocessor.run_full_pipeline(adata)


def run_clustering(adata: sc.AnnData, 
                   resolution: float = 0.5) -> sc.AnnData:
    """
    Convenience function for clustering.
    
    Args:
        adata: Input AnnData object
        resolution: Clustering resolution
        
    Returns:
        AnnData with cluster assignments
    """
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.umap(adata)
    return adata


if __name__ == "__main__":
    # Example usage
    print("ScRNA Preprocessor Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from src.preprocessing import ScRNAPreprocessor")
    print("  preprocessor = ScRNAPreprocessor()")
    print("  adata = preprocessor.run_full_pipeline(adata)")
