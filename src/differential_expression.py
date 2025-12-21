"""
Differential Expression Module for scRNA-seq Analysis Pipeline

This module handles identification of marker genes and differential
expression analysis between cell clusters.

Author: scRNA Analysis Pipeline
License: MIT
"""

import logging
from pathlib import Path
from typing import Union, Optional, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifferentialExpression:
    """
    Differential expression analysis for scRNA-seq data.
    
    This class provides methods for:
    - Identifying marker genes per cluster
    - Comparing expression between groups
    - Visualizing top markers
    
    Example:
        >>> de = DifferentialExpression()
        >>> adata = de.find_marker_genes(adata)
        >>> markers_df = de.get_top_markers(adata, n_genes=20)
    """
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize differential expression analyzer.
        
        Args:
            results_dir: Directory for saving results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(self.results_dir)
    
    def find_marker_genes(self, 
                          adata: sc.AnnData,
                          groupby: str = 'leiden',
                          method: str = 'wilcoxon',
                          min_cells: int = 10,
                          use_raw: bool = False) -> sc.AnnData:
        """
        Identify marker genes for each cluster.
        
        Args:
            adata: Input AnnData object
            groupby: Column in adata.obs to group by
            method: Statistical test ('wilcoxon', 't-test', 'logreg')
            min_cells: Minimum cells for gene to be considered
            use_raw: Whether to use raw counts
            
        Returns:
            AnnData with marker gene results
        """
        logger.info(f"Finding marker genes (method: {method})...")
        
        # Filter genes with very low expression to avoid NaN errors
        if min_cells > 0:
            sc.pp.filter_genes(adata, min_cells=min_cells)
            logger.info(f"  Filtered to {adata.n_vars:,} genes (min_cells={min_cells})")
        
        # Set raw for DE if available
        if adata.raw is None:
            adata.raw = adata
        
        # Run differential expression
        sc.tl.rank_genes_groups(
            adata, 
            groupby=groupby, 
            method=method,
            use_raw=use_raw
        )
        
        # Handle NaN/Inf in log fold changes
        self._clean_de_results(adata)
        
        n_groups = len(adata.obs[groupby].unique())
        logger.info(f"  Found markers for {n_groups} groups")
        
        # Plot top markers
        try:
            sc.pl.rank_genes_groups(
                adata, 
                n_genes=20, 
                sharey=False, 
                save='_marker_genes.png',
                show=False
            )
        except Exception as e:
            logger.warning(f"Could not plot marker genes: {e}")
        
        return adata
    
    def _clean_de_results(self, adata: sc.AnnData) -> None:
        """Clean NaN and Inf values from DE results."""
        if 'rank_genes_groups' not in adata.uns:
            return
        
        for group in adata.uns['rank_genes_groups']['names'].dtype.names:
            df = sc.get.rank_genes_groups_df(adata, group=group)
            
            # Count problematic values
            n_nan = df['logfoldchanges'].isna().sum()
            n_inf = np.isinf(df['logfoldchanges']).sum()
            
            if n_nan > 0 or n_inf > 0:
                logger.debug(f"  Cluster {group}: {n_nan} NaN, {n_inf} Inf values replaced")
    
    def get_top_markers(self, 
                        adata: sc.AnnData,
                        n_genes: int = 10,
                        group: Optional[str] = None) -> pd.DataFrame:
        """
        Get top marker genes as DataFrame.
        
        Args:
            adata: AnnData with rank_genes_groups results
            n_genes: Number of top genes per group
            group: Specific group to get markers for (None for all)
            
        Returns:
            DataFrame with top markers
        """
        if 'rank_genes_groups' not in adata.uns:
            raise ValueError("Run find_marker_genes() first")
        
        if group is not None:
            return sc.get.rank_genes_groups_df(adata, group=group).head(n_genes)
        
        # Get markers for all groups
        all_markers = []
        for g in adata.uns['rank_genes_groups']['names'].dtype.names:
            df = sc.get.rank_genes_groups_df(adata, group=g).head(n_genes)
            df['cluster'] = g
            all_markers.append(df)
        
        return pd.concat(all_markers, ignore_index=True)
    
    def get_markers_dict(self, 
                         adata: sc.AnnData,
                         n_genes: int = 5) -> Dict[str, List[str]]:
        """
        Get top markers as dictionary.
        
        Args:
            adata: AnnData with rank_genes_groups results
            n_genes: Number of top genes per group
            
        Returns:
            Dictionary mapping cluster to gene list
        """
        markers = {}
        
        for group in adata.uns['rank_genes_groups']['names'].dtype.names:
            df = sc.get.rank_genes_groups_df(adata, group=group)
            markers[group] = df['names'].head(n_genes).tolist()
        
        return markers
    
    def save_markers(self, 
                     adata: sc.AnnData,
                     filename: str = 'marker_genes.csv') -> None:
        """
        Save marker genes to CSV file.
        
        Args:
            adata: AnnData with marker gene results
            filename: Output filename
        """
        # Get top markers DataFrame format
        top_markers = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
        
        output_path = self.results_dir / filename
        top_markers.to_csv(output_path, index=False)
        logger.info(f"Saved marker genes to: {output_path}")
    
    def plot_markers_heatmap(self, 
                             adata: sc.AnnData,
                             n_genes: int = 5,
                             groupby: str = 'leiden') -> None:
        """
        Plot heatmap of top marker genes.
        
        Args:
            adata: AnnData with marker results
            n_genes: Number of genes per cluster
            groupby: Grouping column
        """
        try:
            sc.pl.rank_genes_groups_heatmap(
                adata,
                n_genes=n_genes,
                groupby=groupby,
                save='_markers_heatmap.png',
                show=False
            )
            logger.info("Saved marker heatmap")
        except Exception as e:
            logger.warning(f"Could not create heatmap: {e}")
    
    def plot_markers_dotplot(self,
                             adata: sc.AnnData,
                             n_genes: int = 5,
                             groupby: str = 'leiden') -> None:
        """
        Plot dotplot of top marker genes.
        
        Args:
            adata: AnnData with marker results
            n_genes: Number of genes per cluster
            groupby: Grouping column
        """
        try:
            sc.pl.rank_genes_groups_dotplot(
                adata,
                n_genes=n_genes,
                groupby=groupby,
                save='_markers_dotplot.png',
                show=False
            )
            logger.info("Saved marker dotplot")
        except Exception as e:
            logger.warning(f"Could not create dotplot: {e}")
    
    def plot_specific_genes(self,
                            adata: sc.AnnData,
                            genes: List[str],
                            groupby: str = 'leiden') -> None:
        """
        Plot expression of specific genes.
        
        Args:
            adata: AnnData object
            genes: List of gene names to plot
            groupby: Grouping column for violin plot
        """
        # Filter to existing genes
        existing_genes = [g for g in genes if g in adata.var_names]
        
        if not existing_genes:
            logger.warning("None of the specified genes found in data")
            return
        
        try:
            # UMAP colored by gene expression
            sc.pl.umap(
                adata, 
                color=existing_genes, 
                save=f'_gene_expression.png',
                show=False
            )
            
            # Violin plot
            sc.pl.violin(
                adata,
                existing_genes,
                groupby=groupby,
                save='_gene_violin.png',
                show=False
            )
            
            logger.info(f"Saved plots for genes: {existing_genes}")
        except Exception as e:
            logger.warning(f"Could not plot genes: {e}")


def find_markers(adata: sc.AnnData, **kwargs) -> sc.AnnData:
    """
    Convenience function for finding marker genes.
    
    Args:
        adata: Input AnnData object
        **kwargs: Additional arguments
        
    Returns:
        AnnData with marker gene results
    """
    de = DifferentialExpression(**kwargs)
    return de.find_marker_genes(adata)


if __name__ == "__main__":
    print("Differential Expression Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from src.differential_expression import DifferentialExpression")
    print("  de = DifferentialExpression()")
    print("  adata = de.find_marker_genes(adata)")
    print("  markers = de.get_top_markers(adata, n_genes=20)")
