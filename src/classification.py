"""
Machine Learning Classification Module for scRNA-seq Analysis Pipeline

This module implements machine learning models for cell type classification
including Random Forest, SVM, and Deep Neural Networks.

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
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import joblib

# Handle optional imports
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CellTypeClassifier:
    """
    Machine learning classifier for cell type prediction.
    
    This class provides:
    - Random Forest classifier
    - SVM classifier with PCA
    - Deep Neural Network classifier
    - Model evaluation and visualization
    - Feature importance analysis (SHAP)
    
    Example:
        >>> classifier = CellTypeClassifier()
        >>> classifier.train(adata, model_type='random_forest')
        >>> predictions = classifier.predict(new_adata)
        >>> classifier.save_model('models/classifier.pkl')
    """
    
    SUPPORTED_MODELS = ['random_forest', 'svm', 'neural_network', 'all']
    
    def __init__(self, 
                 results_dir: str = 'results/ml_results',
                 models_dir: str = 'models',
                 random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            results_dir: Directory for results and plots
            models_dir: Directory for saving models
            random_state: Random seed for reproducibility
        """
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_state = random_state
        
        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
        
        # Trained models
        self.models = {}
        self.metrics = {}
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
    
    def prepare_data(self, 
                     adata: sc.AnnData,
                     label_col: str = 'cell_type',
                     test_size: float = 0.2,
                     use_smote: bool = True) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            adata: AnnData object
            label_col: Column with cell type labels
            test_size: Proportion for test set
            use_smote: Whether to use SMOTE for class balancing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for classification...")
        
        # Extract expression matrix
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
        
        self.feature_names = X.columns.tolist()
        
        # Get labels
        if label_col not in adata.obs.columns:
            if 'leiden' in adata.obs.columns:
                logger.warning(f"'{label_col}' not found, using 'leiden' clusters")
                label_col = 'leiden'
            else:
                raise ValueError(f"Label column '{label_col}' not found in adata.obs")
        
        y = adata.obs[label_col].copy()
        
        # Handle missing labels
        y = y.dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        logger.info(f"  Data shape: {X.shape}")
        logger.info(f"  Classes: {y.nunique()}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Apply SMOTE if requested
        if use_smote and SMOTE_AVAILABLE:
            logger.info("  Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"  Resampled training shape: {X_train.shape}")
        elif use_smote and not SMOTE_AVAILABLE:
            logger.warning("  SMOTE not available. Install: pip install imbalanced-learn")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store data
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self,
                            n_estimators: int = 100,
                            **kwargs) -> Dict:
        """
        Train Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            **kwargs: Additional RF parameters
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = metrics
        
        return {'model': model, 'metrics': metrics}
    
    def train_svm(self,
                  n_components: int = 50,
                  kernel: str = 'linear',
                  **kwargs) -> Dict:
        """
        Train SVM classifier with PCA.
        
        Args:
            n_components: Number of PCA components
            kernel: SVM kernel type
            **kwargs: Additional SVM parameters
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training SVM with PCA...")
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=min(n_components, self.X_train.shape[1]))
        X_train_pca = self.pca.fit_transform(self.X_train)
        X_test_pca = self.pca.transform(self.X_test)
        
        logger.info(f"  PCA components: {X_train_pca.shape[1]}")
        
        model = SVC(
            kernel=kernel,
            probability=True,
            random_state=self.random_state,
            **kwargs
        )
        
        model.fit(X_train_pca, self.y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_pca)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        self.models['svm'] = model
        self.metrics['svm'] = metrics
        
        return {'model': model, 'metrics': metrics, 'pca': self.pca}
    
    def train_neural_network(self,
                             epochs: int = 50,
                             batch_size: int = 32,
                             **kwargs) -> Dict:
        """
        Train deep neural network classifier.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with model, metrics, and history
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available. Install: pip install tensorflow")
            return None
        
        logger.info("Training Neural Network...")
        
        n_classes = len(np.unique(self.y_train))
        n_features = self.X_train.shape[1]
        
        # One-hot encode labels
        y_train_onehot = keras.utils.to_categorical(self.y_train, n_classes)
        y_test_onehot = keras.utils.to_categorical(self.y_test, n_classes)
        
        # Build model
        model = keras.Sequential([
            layers.Dense(256, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        input_shape=(n_features,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = model.fit(
            self.X_train, y_train_onehot,
            validation_data=(self.X_test, y_test_onehot),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        self.models['neural_network'] = model
        self.metrics['neural_network'] = metrics
        
        # Plot training history
        self._plot_training_history(history)
        
        return {'model': model, 'metrics': metrics, 'history': history}
    
    def train(self, 
              adata: sc.AnnData,
              model_type: str = 'all',
              **kwargs) -> Dict:
        """
        Train classifier(s) on data.
        
        Args:
            adata: AnnData object
            model_type: 'random_forest', 'svm', 'neural_network', or 'all'
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with trained models and metrics
        """
        # Prepare data
        self.prepare_data(adata, **kwargs)
        
        results = {}
        
        if model_type in ['random_forest', 'all']:
            results['random_forest'] = self.train_random_forest()
        
        if model_type in ['svm', 'all']:
            results['svm'] = self.train_svm()
        
        if model_type in ['neural_network', 'all']:
            if TF_AVAILABLE:
                results['neural_network'] = self.train_neural_network()
            else:
                logger.warning("Skipping neural network (TensorFlow not available)")
        
        # Generate summary plots
        self._plot_confusion_matrices()
        self._print_classification_reports()
        
        return results
    
    def predict(self, 
                adata: sc.AnnData,
                model_name: str = 'random_forest') -> np.ndarray:
        """
        Predict cell types for new data.
        
        Args:
            adata: AnnData object
            model_name: Which model to use for prediction
            
        Returns:
            Array of predicted cell types
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        # Extract and scale features
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA if SVM
        if model_name == 'svm' and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Predict
        model = self.models[model_name]
        
        if model_name == 'neural_network':
            y_pred_proba = model.predict(X_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_scaled)
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(y_pred)
        
        return predictions
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def _plot_confusion_matrices(self) -> None:
        """Plot confusion matrices for all trained models."""
        for model_name, metrics in self.metrics.items():
            cm = metrics['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
            plt.tight_layout()
            
            plt.savefig(
                self.results_dir / f'confusion_matrix_{model_name}.png',
                dpi=150
            )
            plt.close()
        
        logger.info("Saved confusion matrix plots")
    
    def _plot_training_history(self, history) -> None:
        """Plot neural network training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=150)
        plt.close()
    
    def _print_classification_reports(self) -> None:
        """Print classification reports for all models."""
        for model_name in self.models:
            model = self.models[model_name]
            
            if model_name == 'svm' and self.pca is not None:
                X_test = self.pca.transform(self.X_test)
            else:
                X_test = self.X_test
            
            if model_name == 'neural_network':
                y_pred_proba = model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test)
            
            print(f"\n{'='*50}")
            print(f"Classification Report - {model_name.replace('_', ' ').title()}")
            print('='*50)
            print(classification_report(
                self.y_test, y_pred,
                target_names=self.label_encoder.classes_
            ))
    
    def compute_shap_values(self, 
                            model_name: str = 'random_forest',
                            n_samples: int = 100) -> None:
        """
        Compute and plot SHAP feature importance.
        
        Args:
            model_name: Model to explain
            n_samples: Number of samples for SHAP
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install: pip install shap")
            return
        
        if model_name not in self.models:
            logger.error(f"Model '{model_name}' not trained")
            return
        
        logger.info(f"Computing SHAP values for {model_name}...")
        
        model = self.models[model_name]
        X_sample = self.X_test[:n_samples]
        
        try:
            if model_name == 'neural_network':
                explainer = shap.GradientExplainer(model, self.X_train[:1000])
            else:
                explainer = shap.TreeExplainer(model) if model_name == 'random_forest' else shap.KernelExplainer(model.predict_proba, self.X_train[:100])
            
            shap_values = explainer.shap_values(X_sample)
            
            # Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names[:X_sample.shape[1]] if self.feature_names else None,
                show=False
            )
            plt.tight_layout()
            plt.savefig(self.results_dir / f'shap_summary_{model_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Saved SHAP summary plot")
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
    
    def save_model(self, filename: str = 'classifier') -> None:
        """
        Save trained models and preprocessors.
        
        Args:
            filename: Base filename for saving
        """
        # Save sklearn models
        for model_name, model in self.models.items():
            if model_name != 'neural_network':
                model_path = self.models_dir / f'{filename}_{model_name}.pkl'
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")
        
        # Save neural network
        if 'neural_network' in self.models and TF_AVAILABLE:
            nn_path = self.models_dir / f'{filename}_neural_network.h5'
            self.models['neural_network'].save(nn_path)
            logger.info(f"Saved neural network to {nn_path}")
        
        # Save preprocessors
        joblib.dump(self.scaler, self.models_dir / f'{filename}_scaler.pkl')
        joblib.dump(self.label_encoder, self.models_dir / f'{filename}_label_encoder.pkl')
        
        if self.pca is not None:
            joblib.dump(self.pca, self.models_dir / f'{filename}_pca.pkl')
        
        logger.info("All models saved successfully")
    
    def load_model(self, filename: str = 'classifier',
                   model_names: List[str] = None) -> None:
        """
        Load trained models and preprocessors.
        
        Args:
            filename: Base filename
            model_names: List of models to load
        """
        if model_names is None:
            model_names = ['random_forest', 'svm', 'neural_network']
        
        # Load preprocessors
        scaler_path = self.models_dir / f'{filename}_scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        encoder_path = self.models_dir / f'{filename}_label_encoder.pkl'
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        
        pca_path = self.models_dir / f'{filename}_pca.pkl'
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
        
        # Load models
        for model_name in model_names:
            if model_name == 'neural_network' and TF_AVAILABLE:
                nn_path = self.models_dir / f'{filename}_neural_network.h5'
                if nn_path.exists():
                    self.models['neural_network'] = keras.models.load_model(nn_path)
                    logger.info(f"Loaded neural network from {nn_path}")
            else:
                model_path = self.models_dir / f'{filename}_{model_name}.pkl'
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} from {model_path}")


if __name__ == "__main__":
    print("Cell Type Classification Module")
    print("=" * 40)
    print(f"\nTensorFlow available: {TF_AVAILABLE}")
    print(f"SMOTE available: {SMOTE_AVAILABLE}")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print("\nUsage:")
    print("  from src.classification import CellTypeClassifier")
    print("  classifier = CellTypeClassifier()")
    print("  classifier.train(adata, model_type='all')")
    print("  predictions = classifier.predict(new_adata)")
