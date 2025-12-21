"""
Data Loader Module for scRNA-seq Analysis Pipeline

This module handles loading 10x Genomics single-cell RNA-seq data
in the filtered feature-barcode matrix format.

Author: scRNA Analysis Pipeline
License: MIT
"""

import os
import gzip
import logging
from pathlib import Path
from typing import Union, Optional, Tuple

import pandas as pd
import numpy as np
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_10x_data(data_path: Union[str, Path],
                  min_genes: int = 0,
                  min_cells: int = 0) -> sc.AnnData:
    """
    Load 10x Genomics filtered feature-barcode matrix data.
    
    This function reads the standard 10x Genomics output format containing:
    - barcodes.tsv.gz: Cell barcodes
    - features.tsv.gz: Gene information
    - matrix.mtx.gz: Sparse expression matrix
    
    Args:
        data_path: Path to directory containing 10x files
        min_genes: Minimum number of genes per cell (filter)
        min_cells: Minimum number of cells per gene (filter)
        
    Returns:
        AnnData object with expression data
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If data format is invalid
    
    Example:
        >>> adata = load_10x_data("data/filtered_feature_bc_matrix")
        >>> print(adata.shape)
        (5000, 20000)
    """
    data_path = Path(data_path)
    
    # Check for required files
    barcode_file = data_path / "barcodes.tsv.gz"
    feature_file = data_path / "features.tsv.gz"
    matrix_file = data_path / "matrix.mtx.gz"
    
    # Also check for uncompressed versions
    if not barcode_file.exists():
        barcode_file = data_path / "barcodes.tsv"
    if not feature_file.exists():
        feature_file = data_path / "features.tsv"
    if not matrix_file.exists():
        matrix_file = data_path / "matrix.mtx"
    
    # Validate files exist
    for f in [barcode_file, feature_file, matrix_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")
    
    logger.info(f"Loading 10x data from: {data_path}")
    
    # Read barcodes (cell IDs)
    logger.info("Reading barcodes...")
    barcodes = _read_tsv(barcode_file, header=None)
    cell_names = barcodes[0].values
    logger.info(f"  Found {len(cell_names):,} cells")
    
    # Read features (genes)
    logger.info("Reading features/genes...")
    features = _read_tsv(feature_file, header=None)
    gene_ids = features[0].values
    gene_names = features[1].values if features.shape[1] > 1 else gene_ids
    logger.info(f"  Found {len(gene_names):,} genes")
    
    # Read expression matrix
    logger.info("Reading expression matrix...")
    matrix = mmread(str(matrix_file))
    matrix = csr_matrix(matrix.T)  # Transpose: cells x genes
    logger.info(f"  Matrix shape: {matrix.shape}")
    
    # Create AnnData object
    adata = sc.AnnData(X=matrix)
    adata.obs_names = cell_names
    adata.var_names = gene_names
    
    # Store gene IDs in var
    adata.var['gene_ids'] = gene_ids
    
    # Make gene names unique
    adata.var_names_make_unique()
    
    # Apply basic filters if specified
    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        logger.info(f"  After min_genes filter: {adata.n_obs:,} cells")
    
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
        logger.info(f"  After min_cells filter: {adata.n_vars:,} genes")
    
    logger.info(f"Loaded AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    return adata


def _read_tsv(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Read TSV file, handling gzipped files automatically.
    
    Args:
        filepath: Path to TSV file (can be .gz)
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame with file contents
    """
    if str(filepath).endswith('.gz'):
        return pd.read_csv(filepath, sep='\t', compression='gzip', **kwargs)
    else:
        return pd.read_csv(filepath, sep='\t', **kwargs)


def load_h5ad(filepath: Union[str, Path]) -> sc.AnnData:
    """
    Load preprocessed AnnData from H5AD file.
    
    Args:
        filepath: Path to .h5ad file
        
    Returns:
        AnnData object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"H5AD file not found: {filepath}")
    
    logger.info(f"Loading H5AD file: {filepath}")
    adata = sc.read_h5ad(filepath)
    logger.info(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    
    return adata


def save_h5ad(adata: sc.AnnData, 
              filepath: Union[str, Path],
              compression: str = 'gzip') -> None:
    """
    Save AnnData object to H5AD file.
    
    Args:
        adata: AnnData object to save
        filepath: Output file path
        compression: Compression method
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving AnnData to: {filepath}")
    adata.write_h5ad(filepath, compression=compression)
    logger.info("Save complete")


def extract_expression_matrix(adata: sc.AnnData) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract expression matrix and cell labels from AnnData.
    
    Args:
        adata: AnnData object
        
    Returns:
        Tuple of (expression DataFrame, cell type Series)
    """
    # Get expression data
    if hasattr(adata.X, 'toarray'):
        X = pd.DataFrame(
            adata.X.toarray(),
            index=adata.obs_names,
            columns=adata.var_names
        )
    else:
        X = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
    
    # Get cell type labels if available
    if 'cell_type' in adata.obs.columns:
        y = adata.obs['cell_type']
    elif 'leiden' in adata.obs.columns:
        y = adata.obs['leiden']
    else:
        y = pd.Series(['Unknown'] * adata.n_obs, index=adata.obs_names)
    
    return X, y


def validate_10x_directory(data_path: Union[str, Path]) -> bool:
    """
    Validate that a directory contains required 10x Genomics files.
    
    Args:
        data_path: Path to check
        
    Returns:
        True if valid 10x directory
    """
    data_path = Path(data_path)
    
    required_files = [
        ('barcodes.tsv.gz', 'barcodes.tsv'),
        ('features.tsv.gz', 'features.tsv', 'genes.tsv.gz', 'genes.tsv'),
        ('matrix.mtx.gz', 'matrix.mtx')
    ]
    
    for file_options in required_files:
        found = False
        for filename in file_options:
            if (data_path / filename).exists():
                found = True
                break
        if not found:
            return False
    
    return True


def get_data_info(data_path: Union[str, Path]) -> dict:
    """
    Get information about 10x dataset without fully loading it.
    
    Args:
        data_path: Path to 10x data directory
        
    Returns:
        Dictionary with dataset information
    """
    data_path = Path(data_path)
    
    info = {
        'path': str(data_path),
        'valid': validate_10x_directory(data_path),
        'files': []
    }
    
    if not info['valid']:
        return info
    
    # Get file sizes
    for f in data_path.iterdir():
        if f.suffix in ['.gz', '.tsv', '.mtx']:
            info['files'].append({
                'name': f.name,
                'size_mb': f.stat().st_size / (1024 * 1024)
            })
    
    # Quick count of cells and genes
    barcode_file = data_path / "barcodes.tsv.gz"
    if not barcode_file.exists():
        barcode_file = data_path / "barcodes.tsv"
    
    if barcode_file.exists():
        barcodes = _read_tsv(barcode_file, header=None)
        info['n_cells'] = len(barcodes)
    
    feature_file = data_path / "features.tsv.gz"
    if not feature_file.exists():
        feature_file = data_path / "features.tsv"
    
    if feature_file.exists():
        features = _read_tsv(feature_file, header=None)
        info['n_genes'] = len(features)
    
    return info


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
        # Validate and show info
        info = get_data_info(data_path)
        
        print("\n=== 10x Genomics Data Info ===")
        print(f"Path: {info['path']}")
        print(f"Valid: {info['valid']}")
        
        if info['valid']:
            print(f"Cells: {info.get('n_cells', 'Unknown'):,}")
            print(f"Genes: {info.get('n_genes', 'Unknown'):,}")
            print("\nFiles:")
            for f in info['files']:
                print(f"  {f['name']}: {f['size_mb']:.2f} MB")
            
            # Load data
            print("\nLoading data...")
            adata = load_10x_data(data_path)
            print(f"Loaded: {adata}")
    else:
        print("Usage: python data_loader.py <path_to_10x_data>")
