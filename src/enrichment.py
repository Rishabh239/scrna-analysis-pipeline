"""
Enrichment Analysis Module for scRNA-seq Analysis Pipeline

This module handles Gene Ontology (GO) and KEGG pathway enrichment
analysis for marker genes from single-cell clusters.

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
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if gseapy is available
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    logger.warning("gseapy not installed. Enrichment analysis will be limited.")


class EnrichmentAnalyzer:
    """
    Gene enrichment analysis for scRNA-seq data.
    
    This class provides methods for:
    - GO (Gene Ontology) enrichment analysis
    - KEGG pathway enrichment analysis
    - Cross-cluster enrichment comparison
    - Visualization of enrichment results
    
    Example:
        >>> analyzer = EnrichmentAnalyzer()
        >>> go_results = analyzer.run_go_enrichment(gene_list)
        >>> analyzer.plot_enrichment(go_results, 'GO')
    """
    
    # Available gene sets
    GO_GENE_SETS = [
        'GO_Biological_Process_2021',
        'GO_Molecular_Function_2021',
        'GO_Cellular_Component_2021'
    ]
    
    KEGG_GENE_SETS = [
        'KEGG_2021_Human',
        'KEGG_2019_Human'
    ]
    
    def __init__(self, 
                 results_dir: str = 'results',
                 organism: str = 'human'):
        """
        Initialize enrichment analyzer.
        
        Args:
            results_dir: Directory for saving results
            organism: Organism for enrichment ('human' or 'mouse')
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.organism = organism
        
        if not GSEAPY_AVAILABLE:
            logger.warning("gseapy not available. Install with: pip install gseapy")
    
    def extract_marker_genes(self, 
                             adata: sc.AnnData,
                             n_genes: int = 100,
                             cluster: Optional[str] = None) -> List[str]:
        """
        Extract top marker genes from AnnData.
        
        Args:
            adata: AnnData with rank_genes_groups results
            n_genes: Number of top genes to extract
            cluster: Specific cluster (None for all clusters)
            
        Returns:
            List of gene names
        """
        if 'rank_genes_groups' not in adata.uns:
            raise ValueError("Run differential expression analysis first")
        
        gene_names = adata.uns['rank_genes_groups']['names']
        
        if cluster is not None:
            # Get genes for specific cluster
            genes = list(gene_names[cluster][:n_genes])
        else:
            # Get genes from all clusters
            genes = []
            for col in gene_names.dtype.names:
                genes.extend(list(gene_names[col][:n_genes // len(gene_names.dtype.names)]))
            genes = list(set(genes))[:n_genes]
        
        logger.info(f"Extracted {len(genes)} marker genes")
        return genes
    
    def run_go_enrichment(self,
                          gene_list: List[str],
                          gene_set: str = 'GO_Biological_Process_2021') -> Optional[pd.DataFrame]:
        """
        Run Gene Ontology enrichment analysis.
        
        Args:
            gene_list: List of gene symbols
            gene_set: GO gene set to use
            
        Returns:
            DataFrame with enrichment results
        """
        if not GSEAPY_AVAILABLE:
            logger.error("gseapy required for enrichment analysis")
            return None
        
        logger.info(f"Running GO enrichment ({gene_set})...")
        
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_set,
                organism=self.organism,
                outdir=None,
                no_plot=True
            )
            
            results = enr.results
            logger.info(f"  Found {len(results)} enriched terms")
            
            return results
            
        except Exception as e:
            logger.error(f"GO enrichment failed: {e}")
            return None
    
    def run_kegg_enrichment(self,
                            gene_list: List[str],
                            gene_set: str = 'KEGG_2021_Human') -> Optional[pd.DataFrame]:
        """
        Run KEGG pathway enrichment analysis.
        
        Args:
            gene_list: List of gene symbols
            gene_set: KEGG gene set to use
            
        Returns:
            DataFrame with enrichment results
        """
        if not GSEAPY_AVAILABLE:
            logger.error("gseapy required for enrichment analysis")
            return None
        
        logger.info(f"Running KEGG enrichment ({gene_set})...")
        
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_set,
                organism=self.organism,
                outdir=None,
                no_plot=True
            )
            
            results = enr.results
            logger.info(f"  Found {len(results)} enriched pathways")
            
            return results
            
        except Exception as e:
            logger.error(f"KEGG enrichment failed: {e}")
            return None
    
    def run_enrichment_all_clusters(self,
                                    adata: sc.AnnData,
                                    n_genes: int = 100) -> Dict[str, Dict]:
        """
        Run enrichment analysis for all clusters.
        
        Args:
            adata: AnnData with marker genes
            n_genes: Number of genes per cluster
            
        Returns:
            Dictionary with results for each cluster
        """
        results = {}
        
        clusters = adata.uns['rank_genes_groups']['names'].dtype.names
        
        for cluster in clusters:
            logger.info(f"Processing cluster {cluster}...")
            
            genes = self.extract_marker_genes(adata, n_genes=n_genes, cluster=cluster)
            
            cluster_results = {
                'genes': genes,
                'go': self.run_go_enrichment(genes),
                'kegg': self.run_kegg_enrichment(genes)
            }
            
            results[cluster] = cluster_results
            
            # Save individual results
            if cluster_results['go'] is not None:
                cluster_results['go'].to_csv(
                    self.results_dir / f'GO_enrichment_cluster_{cluster}.csv',
                    index=False
                )
            
            if cluster_results['kegg'] is not None:
                cluster_results['kegg'].to_csv(
                    self.results_dir / f'KEGG_enrichment_cluster_{cluster}.csv',
                    index=False
                )
        
        return results
    
    def plot_enrichment_bar(self,
                            results: pd.DataFrame,
                            title: str = 'Enrichment Analysis',
                            top_n: int = 10,
                            filename: Optional[str] = None) -> None:
        """
        Plot enrichment results as horizontal bar chart.
        
        Args:
            results: Enrichment results DataFrame
            title: Plot title
            top_n: Number of top terms to show
            filename: Output filename (None for auto)
        """
        if results is None or len(results) == 0:
            logger.warning("No results to plot")
            return
        
        # Get top results by adjusted p-value
        top_results = results.sort_values('Adjusted P-value').head(top_n)
        
        if len(top_results) == 0:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate -log10(p-value)
        y_values = -np.log10(top_results['Adjusted P-value'].astype(float) + 1e-300)
        
        # Truncate long term names
        terms = [t[:50] + '...' if len(t) > 50 else t for t in top_results['Term']]
        
        # Bar plot
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(terms)))
        ax.barh(range(len(terms)), y_values, color=colors)
        
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms)
        ax.set_xlabel('-log10(Adjusted P-value)')
        ax.set_title(title)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save
        if filename is None:
            filename = title.lower().replace(' ', '_') + '.png'
        
        plt.savefig(self.results_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved enrichment plot: {filename}")
    
    def create_enrichment_heatmap(self,
                                  cluster_results: Dict[str, Dict],
                                  analysis_type: str = 'go',
                                  top_n: int = 5,
                                  filename: Optional[str] = None) -> None:
        """
        Create heatmap of enrichment across clusters.
        
        Args:
            cluster_results: Results from run_enrichment_all_clusters
            analysis_type: 'go' or 'kegg'
            top_n: Number of top terms per cluster
            filename: Output filename
        """
        # Collect top terms from each cluster
        all_terms = set()
        
        for cluster, data in cluster_results.items():
            df = data.get(analysis_type)
            if df is not None and len(df) > 0:
                df['Adjusted P-value'] = pd.to_numeric(df['Adjusted P-value'], errors='coerce')
                top_terms = df.sort_values('Adjusted P-value').head(top_n)['Term']
                all_terms.update(top_terms)
        
        if not all_terms:
            logger.warning(f"No {analysis_type.upper()} terms found")
            return
        
        # Create heatmap data
        all_terms = list(all_terms)
        heatmap_data = pd.DataFrame(
            index=all_terms,
            columns=[f'Cluster {c}' for c in cluster_results.keys()]
        )
        
        # Fill with -log10(p-values)
        for cluster, data in cluster_results.items():
            df = data.get(analysis_type)
            if df is not None and len(df) > 0:
                df = df.set_index('Term')
                for term in all_terms:
                    if term in df.index:
                        pval = pd.to_numeric(df.loc[term, 'Adjusted P-value'], errors='coerce')
                        heatmap_data.at[term, f'Cluster {cluster}'] = -np.log10(pval + 1e-300)
        
        heatmap_data = heatmap_data.fillna(0).astype(float)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(all_terms) * 0.4)))
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='coolwarm',
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_xlabel('Clusters')
        ax.set_ylabel(f'{analysis_type.upper()} Terms')
        ax.set_title(f'{analysis_type.upper()} Enrichment Across Clusters')
        
        plt.tight_layout()
        
        # Save
        if filename is None:
            filename = f'{analysis_type}_heatmap.png'
        
        plt.savefig(self.results_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heatmap: {filename}")
    
    def save_results(self,
                     results: pd.DataFrame,
                     filename: str) -> None:
        """
        Save enrichment results to CSV.
        
        Args:
            results: Enrichment DataFrame
            filename: Output filename
        """
        if results is not None:
            output_path = self.results_dir / filename
            results.to_csv(output_path, index=False)
            logger.info(f"Saved results to: {output_path}")


def run_enrichment(adata: sc.AnnData, 
                   results_dir: str = 'results',
                   n_genes: int = 100) -> Dict:
    """
    Convenience function to run full enrichment analysis.
    
    Args:
        adata: AnnData with marker genes
        results_dir: Output directory
        n_genes: Number of genes per cluster
        
    Returns:
        Dictionary with all enrichment results
    """
    analyzer = EnrichmentAnalyzer(results_dir=results_dir)
    
    # Run enrichment for all clusters
    results = analyzer.run_enrichment_all_clusters(adata, n_genes=n_genes)
    
    # Create summary heatmaps
    analyzer.create_enrichment_heatmap(results, 'go')
    analyzer.create_enrichment_heatmap(results, 'kegg')
    
    return results


if __name__ == "__main__":
    print("Enrichment Analysis Module")
    print("=" * 40)
    print(f"\ngseapy available: {GSEAPY_AVAILABLE}")
    print("\nUsage:")
    print("  from src.enrichment import EnrichmentAnalyzer")
    print("  analyzer = EnrichmentAnalyzer()")
    print("  go_results = analyzer.run_go_enrichment(gene_list)")
    print("  analyzer.plot_enrichment_bar(go_results)")
