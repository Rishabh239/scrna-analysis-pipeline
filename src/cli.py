"""
Command-Line Interface for scRNA-seq Analysis Pipeline

This module provides CLI commands for running the analysis pipeline.

Author: scRNA Analysis Pipeline
License: MIT
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default cell type mapping (extend as needed based on your data)
DEFAULT_CELL_TYPES = {
    "0": "T cells",
    "1": "B cells",
    "2": "Macrophages",
    "3": "Dendritic cells",
    "4": "NK cells",
    "5": "Monocytes",
    "6": "Plasma cells",
    "7": "Progenitor cells",
    "8": "Erythrocytes",
    "9": "Platelets",
    "10": "Unknown_10",
    "11": "Unknown_11",
    "12": "Unknown_12"
}


def run_full_pipeline(args):
    """Run the complete analysis pipeline."""
    from .data_loader import load_10x_data, save_h5ad
    from .preprocessing import ScRNAPreprocessor
    from .differential_expression import DifferentialExpression
    from .enrichment import EnrichmentAnalyzer
    from .classification import CellTypeClassifier
    
    logger.info("=" * 50)
    logger.info("Starting Full scRNA-seq Analysis Pipeline")
    logger.info("=" * 50)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    logger.info("\n[1/6] Loading data...")
    adata = load_10x_data(args.input)
    
    # Step 2: Preprocessing
    logger.info("\n[2/6] Preprocessing...")
    preprocessor = ScRNAPreprocessor(
        results_dir=str(output_dir / 'preprocessing'),
        qc_params={
            'min_genes': args.min_genes,
            'max_counts': args.max_counts,
            'max_mt_pct': args.max_mt_pct
        }
    )
    
    adata = preprocessor.run_full_pipeline(
        adata,
        cell_type_dict=DEFAULT_CELL_TYPES if args.annotate else None
    )
    
    # Step 3: Differential expression
    logger.info("\n[3/6] Finding marker genes...")
    de = DifferentialExpression(results_dir=str(output_dir / 'differential_expression'))
    adata = de.find_marker_genes(adata)
    de.save_markers(adata)
    
    # Step 4: Enrichment analysis
    logger.info("\n[4/6] Running enrichment analysis...")
    enrichment = EnrichmentAnalyzer(results_dir=str(output_dir / 'enrichment'))
    try:
        # Run GO and KEGG enrichment for all clusters
        cluster_results = enrichment.run_enrichment_all_clusters(adata, n_genes=100)
        
        # Create cross-cluster heatmaps
        logger.info("  Creating GO heatmap...")
        enrichment.create_enrichment_heatmap(cluster_results, 'go', filename='go_heatmap.png')
        
        logger.info("  Creating KEGG heatmap...")
        enrichment.create_enrichment_heatmap(cluster_results, 'kegg', filename='kegg_heatmap.png')
        
        # Plot individual enrichment bar charts for each cluster
        for cluster_id, results in cluster_results.items():
            if results['go'] is not None and len(results['go']) > 0:
                enrichment.plot_enrichment_bar(
                    results['go'], 
                    title=f'GO Enrichment (Cluster {cluster_id})',
                    filename=f'go_enrichment_cluster_{cluster_id}.png'
                )
            if results['kegg'] is not None and len(results['kegg']) > 0:
                enrichment.plot_enrichment_bar(
                    results['kegg'],
                    title=f'KEGG Enrichment (Cluster {cluster_id})',
                    filename=f'kegg_enrichment_cluster_{cluster_id}.png'
                )
        
        logger.info("  Enrichment analysis complete!")
        logger.info(f"  Results saved to: {output_dir / 'enrichment'}")
    except Exception as e:
        logger.warning(f"Enrichment analysis failed: {e}")
        logger.warning("  Install gseapy: pip install gseapy")
    
    # Step 5: Classification
    if args.classify:
        logger.info("\n[5/6] Training classifiers...")
        classifier = CellTypeClassifier(
            results_dir=str(output_dir / 'classification'),
            models_dir=str(output_dir / 'models')
        )
        classifier.train(adata, model_type=args.model_type)
        classifier.save_model('cell_classifier')
    else:
        logger.info("\n[5/6] Skipping classification (use --classify to enable)")
    
    # Step 6: Save results
    logger.info("\n[6/6] Saving results...")
    save_h5ad(adata, output_dir / 'processed_data.h5ad')
    
    logger.info("\n" + "=" * 50)
    logger.info("Pipeline Complete!")
    logger.info("=" * 50)
    logger.info(f"Results saved to: {output_dir}")


def run_preprocess(args):
    """Run only preprocessing."""
    from .data_loader import load_10x_data, save_h5ad
    from .preprocessing import ScRNAPreprocessor
    
    logger.info("Running preprocessing...")
    
    # Load data
    adata = load_10x_data(args.input)
    
    # Preprocess
    output_dir = Path(args.output)
    preprocessor = ScRNAPreprocessor(results_dir=str(output_dir))
    adata = preprocessor.run_full_pipeline(adata)
    
    # Save
    save_h5ad(adata, output_dir / 'preprocessed.h5ad')
    logger.info(f"Saved preprocessed data to: {output_dir}")


def run_classify(args):
    """Run classification on preprocessed data."""
    from .data_loader import load_h5ad
    from .classification import CellTypeClassifier
    
    logger.info("Running classification...")
    
    # Load preprocessed data
    adata = load_h5ad(args.input)
    
    # Train classifiers
    output_dir = Path(args.output)
    classifier = CellTypeClassifier(
        results_dir=str(output_dir),
        models_dir=str(output_dir / 'models')
    )
    
    classifier.train(adata, model_type=args.model_type)
    classifier.save_model('cell_classifier')
    
    logger.info(f"Saved models to: {output_dir / 'models'}")


def run_predict(args):
    """Predict cell types using trained model."""
    from .data_loader import load_h5ad
    from .classification import CellTypeClassifier
    
    logger.info("Running prediction...")
    
    # Load data
    adata = load_h5ad(args.input)
    
    # Load model
    classifier = CellTypeClassifier(models_dir=args.model_dir)
    classifier.load_model('cell_classifier', model_names=[args.model_type])
    
    # Predict
    predictions = classifier.predict(adata, model_name=args.model_type)
    
    # Save predictions
    adata.obs['predicted_cell_type'] = predictions
    
    output_path = Path(args.output)
    adata.obs[['predicted_cell_type']].to_csv(output_path)
    
    logger.info(f"Saved predictions to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="scRNA-seq Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m src.cli run --input data/filtered_feature_bc_matrix --output results/
  
  # Run only preprocessing
  python -m src.cli preprocess --input data/filtered_feature_bc_matrix --output results/
  
  # Run classification on preprocessed data
  python -m src.cli classify --input results/preprocessed.h5ad --output results/
  
  # Predict cell types
  python -m src.cli predict --input new_data.h5ad --model-dir models/ --output predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run full pipeline
    run_parser = subparsers.add_parser('run', help='Run complete pipeline')
    run_parser.add_argument('--input', '-i', required=True, help='Path to 10x data directory')
    run_parser.add_argument('--output', '-o', default='results', help='Output directory')
    run_parser.add_argument('--min-genes', type=int, default=200, help='Min genes per cell')
    run_parser.add_argument('--max-counts', type=int, default=25000, help='Max counts per cell')
    run_parser.add_argument('--max-mt-pct', type=float, default=5.0, help='Max mitochondrial %%')
    run_parser.add_argument('--annotate', action='store_true', help='Auto-annotate cell types')
    run_parser.add_argument('--enrichment', action='store_true', help='Run enrichment analysis')
    run_parser.add_argument('--classify', action='store_true', help='Train classifiers')
    run_parser.add_argument('--model-type', default='all', 
                           choices=['random_forest', 'svm', 'neural_network', 'all'],
                           help='ML model type to train')
    
    # Preprocess only
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocessing only')
    preprocess_parser.add_argument('--input', '-i', required=True, help='Path to 10x data')
    preprocess_parser.add_argument('--output', '-o', default='results', help='Output directory')
    
    # Classify only
    classify_parser = subparsers.add_parser('classify', help='Train classifiers')
    classify_parser.add_argument('--input', '-i', required=True, help='Path to preprocessed h5ad')
    classify_parser.add_argument('--output', '-o', default='results', help='Output directory')
    classify_parser.add_argument('--model-type', default='all',
                                choices=['random_forest', 'svm', 'neural_network', 'all'],
                                help='Model type')
    
    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict cell types')
    predict_parser.add_argument('--input', '-i', required=True, help='Path to h5ad file')
    predict_parser.add_argument('--model-dir', '-m', required=True, help='Directory with trained models')
    predict_parser.add_argument('--model-type', default='random_forest', help='Model to use')
    predict_parser.add_argument('--output', '-o', default='predictions.csv', help='Output file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    commands = {
        'run': run_full_pipeline,
        'preprocess': run_preprocess,
        'classify': run_classify,
        'predict': run_predict
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
