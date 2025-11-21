"""
Decision Tree and Random Forest Analysis for Speed Dating Match Prediction

This module implements a comprehensive machine learning pipeline to predict speed dating matches
using Decision Tree Classifiers and Random Forest Classifiers. It follows SOLID, KISS, and DRY
principles with robust error handling, logging, and visualization capabilities.

Author: AIDA Project
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
import joblib
import logging

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure warnings and display
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DataPreprocessor:
    """
    Handles all data loading and preprocessing operations.
    Single Responsibility: Data preparation and feature engineering.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.feature_names = None
        self.target_name = 'match'
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
        """Load cleaned dataset."""
        self.logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        self.logger.info(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def select_features(self) -> List[str]:
        """
        Select relevant features for modeling.
        
        Categories:
        - Demographic features
        - Racial features
        - Attribute ratings
        - Partner ratings
        - Preference features
        - Interest scores
        - Derived features
        - Behavioral features
        """
        # Define feature categories
        demographic_features = ['gender', 'age', 'age_o', 'age_diff']
        
        # Racial features
        racial_features = ['samerace'] + [col for col in self.data.columns if col.startswith('race_')]
        
        # Attribute ratings (self)
        attribute_features = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
        
        # Partner ratings
        partner_features = ['attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o']
        
        # Preference features
        preference_features = ['pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha']
        
        # Interest scores
        interest_features = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 
                           'art', 'hiking', 'gaming', 'clubbing', 'reading', 
                           'tv', 'theater', 'movies', 'concerts', 'music', 
                           'shopping', 'yoga']
        
        # Derived features
        derived_features = ['attr_diff', 'sinc_diff', 'intel_diff', 'fun_diff', 
                          'amb_diff', 'shar_diff', 'preference_match_score',
                          'avg_rating_given', 'avg_rating_received', 'rating_asymmetry']
        
        # Behavioral features
        behavioral_features = ['goal', 'date', 'go_out', 'exphappy', 'expnum']
        
        # Combine all features
        all_features = (demographic_features + racial_features + attribute_features + 
                       partner_features + preference_features + interest_features +
                       derived_features + behavioral_features)
        
        # Filter to only include features that exist in the dataset
        available_features = [f for f in all_features if f in self.data.columns]
        
        self.feature_names = available_features
        self.logger.info(f"Selected {len(available_features)} features for modeling")
        
        return available_features
    
    def prepare_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for modeling.
        
        Returns:
            X: Features DataFrame
            y: Target Series
        """
        if self.data is None:
            self.load_data()
        
        if self.feature_names is None:
            self.select_features()
        
        # Ensure target exists
        if self.target_name not in self.data.columns:
            raise ValueError(f"Target variable '{self.target_name}' not found in dataset")
        
        # Extract features and target
        X = self.data[self.feature_names].copy()
        y = self.data[self.target_name].copy()
        
        # Handle missing values (forward fill, then median)
        X = X.fillna(method='ffill').fillna(X.median())
        
        # Convert boolean to int if needed
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
        
        self.logger.info(f"Dataset prepared: X shape {X.shape}, y shape {y.shape}")
        self.logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
        
        return X, y


class ModelTrainer:
    """
    Handles model training operations.
    Single Responsibility: Model training and hyperparameter tuning.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Stratified train-test split."""
        self.logger.info(f"Splitting data with test_size={test_size}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        self.feature_names = X.columns.tolist()
        
        self.logger.info(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        self.logger.info(f"Train target distribution:\n{self.y_train.value_counts(normalize=True)}")
        self.logger.info(f"Test target distribution:\n{self.y_test.value_counts(normalize=True)}")
    
    def handle_imbalance(self, method: str = 'smote'):
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            method: Method to handle imbalance ('smote' or 'none')
        """
        if method == 'smote':
            self.logger.info("Applying SMOTE to handle class imbalance")
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            self.logger.info(f"After SMOTE - Train set: {self.X_train.shape}")
            self.logger.info(f"After SMOTE - Target distribution:\n{pd.Series(self.y_train).value_counts(normalize=True)}")
        else:
            self.logger.info("No imbalance handling applied")
    
    def train_decision_tree_baseline(self) -> DecisionTreeClassifier:
        """Train baseline decision tree classifier."""
        self.logger.info("Training baseline Decision Tree")
        
        dt_model = DecisionTreeClassifier(
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        dt_model.fit(self.X_train, self.y_train)
        
        self.logger.info("Baseline Decision Tree trained")
        return dt_model
    
    def tune_decision_tree(self) -> Tuple[DecisionTreeClassifier, Dict]:
        """Hyperparameter tuning using GridSearchCV."""
        self.logger.info("Starting Decision Tree hyperparameter tuning")
        
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        }
        
        dt = DecisionTreeClassifier(
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        grid_search = GridSearchCV(
            dt, param_grid, cv=5, 
            scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_random_forest(self, tune: bool = True) -> RandomForestClassifier:
        """Train Random Forest classifier with optional tuning."""
        self.logger.info(f"Training Random Forest (tune={tune})")
        
        if tune:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=5,
                scoring='f1', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            rf.fit(self.X_train, self.y_train)
            
            self.logger.info("Random Forest trained (no tuning)")
            return rf


class ModelEvaluator:
    """
    Handles model evaluation and comparison.
    Single Responsibility: Model evaluation and metrics calculation.
    """
    
    def __init__(self, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model, model_name: str) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.logger.info(f"Evaluating {model_name}")
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'avg_precision': average_precision_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = metrics
        
        self.logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"{model_name} - F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Create comparison table."""
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'Avg Precision': metrics['avg_precision']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        self.logger.info("\nModel Comparison:\n" + comparison_df.to_string())
        
        return comparison_df


class Visualizer:
    """
    Handles all visualization tasks.
    Single Responsibility: Visualization generation and export.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib and seaborn style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("notebook", font_scale=1.2)
        sns.set_palette("Set2")
    
    def plot_decision_tree(self, model, feature_names: List[str], max_depth: int = 3):
        """Visualize decision tree structure."""
        self.logger.info(f"Plotting decision tree (max_depth={max_depth})")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, class_names=['No Match', 'Match'],
                 filled=True, rounded=True, fontsize=10, max_depth=max_depth, ax=ax)
        
        plt.title(f'Decision Tree Structure (Max Depth: {max_depth})', fontsize=16, pad=20)
        plt.tight_layout()
        
        output_path = self.output_dir / f'decision_tree_structure_depth{max_depth}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Decision tree plot saved to {output_path}")
    
    def plot_feature_importance(self, model, feature_names: List[str], model_name: str, top_n: int = 20):
        """Plot feature importance."""
        self.logger.info(f"Plotting feature importance for {model_name}")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, pad=20)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = self.output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature importance plot saved to {output_path}")
        
        return importances
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot confusion matrix heatmap."""
        self.logger.info(f"Plotting confusion matrix for {model_name}")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Match', 'Match'],
                   yticklabels=['No Match', 'Match'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, pad=20)
        plt.tight_layout()
        
        output_path = self.output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {output_path}")
    
    def plot_roc_curves(self, results_dict: Dict[str, Dict], y_test):
        """Compare ROC curves."""
        self.logger.info("Plotting ROC curves comparison")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, metrics in results_dict.items():
            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
            auc_score = metrics['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'roc_curves_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"ROC curves saved to {output_path}")
    
    def plot_precision_recall_curves(self, results_dict: Dict[str, Dict], y_test):
        """Compare PR curves."""
        self.logger.info("Plotting Precision-Recall curves")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, metrics in results_dict.items():
            precision, recall, _ = precision_recall_curve(y_test, metrics['y_pred_proba'])
            avg_precision = metrics['avg_precision']
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, pad=20)
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'precision_recall_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"PR curves saved to {output_path}")
    
    def create_comparison_dashboard(self, comparison_df: pd.DataFrame):
        """Create interactive comparison dashboard using Plotly."""
        self.logger.info("Creating comparison dashboard")
        
        # Prepare data for radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for idx, row in comparison_df.iterrows():
            values = [row[metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='Model Performance Comparison',
            showlegend=True,
            height=600
        )
        
        output_path = self.output_dir / 'model_comparison_dashboard.html'
        fig.write_html(str(output_path))
        
        self.logger.info(f"Comparison dashboard saved to {output_path}")


class ReportGenerator:
    """
    Generates comprehensive analysis reports.
    Single Responsibility: Report generation and results export.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_markdown_report(self, results: Dict, comparison_df: pd.DataFrame, 
                                 dt_params: Dict, feature_importance_dt: np.ndarray,
                                 feature_importance_rf: np.ndarray, feature_names: List[str]) -> str:
        """Generate comprehensive markdown report."""
        self.logger.info("Generating analysis report")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Decision Tree & Random Forest Analysis Report
## Speed Dating Match Prediction

**Generated:** {timestamp}

---

## Executive Summary

This report presents a comprehensive analysis of speed dating match prediction using Decision Tree and Random Forest classifiers. The models were trained on a cleaned dataset with {len(feature_names)} features encompassing demographic, behavioral, preference, and derived attributes.

### Key Findings

"""
        
        # Add model comparison
        report += "\n### Model Performance Comparison\n\n"
        report += comparison_df.to_markdown(index=False)
        report += "\n\n"
        
        # Best model
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'F1-Score']
        best_auc = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'ROC-AUC']
        
        report += f"""
**Best Performing Model:** {best_model}
- **F1-Score:** {best_f1:.4f}
- **ROC-AUC:** {best_auc:.4f}

---

## Model Configurations

### Decision Tree
"""
        
        report += "\n**Optimized Hyperparameters:**\n"
        for param, value in dt_params.items():
            report += f"- `{param}`: {value}\n"
        
        report += "\n### Random Forest\n"
        report += "- Random Forest was optimized using GridSearchCV with 5-fold cross-validation\n"
        report += "- Uses ensemble of decision trees for improved generalization\n"
        
        # Feature importance
        report += "\n---\n\n## Feature Importance Analysis\n\n"
        report += "### Top 10 Most Important Features (Random Forest)\n\n"
        
        # Get top 10 features
        rf_importance_indices = np.argsort(feature_importance_rf)[::-1][:10]
        
        report += "| Rank | Feature | Importance |\n"
        report += "|------|---------|------------|\n"
        
        for rank, idx in enumerate(rf_importance_indices, 1):
            report += f"| {rank} | {feature_names[idx]} | {feature_importance_rf[idx]:.4f} |\n"
        
        # Add insights
        report += "\n---\n\n## Key Insights\n\n"
        
        for model_name, metrics in results.items():
            report += f"\n### {model_name}\n\n"
            report += f"**Performance Metrics:**\n"
            report += f"- Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"- Precision: {metrics['precision']:.4f}\n"
            report += f"- Recall: {metrics['recall']:.4f}\n"
            report += f"- F1-Score: {metrics['f1']:.4f}\n"
            report += f"- ROC-AUC: {metrics['roc_auc']:.4f}\n\n"
        
        report += "\n---\n\n## Recommendations\n\n"
        report += f"1. **Deployment:** Use the {best_model} for production deployment\n"
        report += "2. **Feature Engineering:** Focus on the top importance features for model optimization\n"
        report += "3. **Model Monitoring:** Continuously monitor model performance and retrain as needed\n"
        report += "4. **Ensemble Methods:** Consider combining models for improved predictions\n"
        
        report += "\n---\n\n## Conclusion\n\n"
        report += f"The {best_model} achieved the best performance with an F1-score of {best_f1:.4f} "
        report += f"and ROC-AUC of {best_auc:.4f}. This indicates strong predictive capability for "
        report += "speed dating match outcomes based on the selected features.\n"
        
        # Save report
        report_path = self.output_dir / 'decision_tree_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to {report_path}")
        
        return report
    
    def export_results(self, models: Dict, metrics: Dict, comparison_df: pd.DataFrame,
                      feature_names: List[str]):
        """Export all results to files."""
        self.logger.info("Exporting results")
        
        # Create data directory
        data_dir = self.output_dir.parent / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison metrics
        comparison_df.to_csv(data_dir / 'model_comparison_metrics.csv', index=False)
        
        # Save feature importances
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                importance_df.to_csv(data_dir / filename, index=False)
        
        self.logger.info(f"Results exported to {data_dir}")
    
    def save_models(self, models: Dict):
        """Save trained models."""
        self.logger.info("Saving models")
        
        models_dir = self.output_dir.parent / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            model_path = models_dir / filename
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved to {model_path}")
    
    def export_decision_rules(self, model, feature_names: List[str]):
        """Export decision rules from tree."""
        self.logger.info("Exporting decision rules")
        
        tree_rules = export_text(model, feature_names=feature_names)
        
        rules_path = self.output_dir / 'decision_rules.txt'
        with open(rules_path, 'w') as f:
            f.write("DECISION TREE RULES\n")
            f.write("=" * 80 + "\n\n")
            f.write(tree_rules)
        
        self.logger.info(f"Decision rules saved to {rules_path}")


class DecisionTreeAnalyzer:
    """
    Main orchestrator for decision tree analysis.
    Coordinates all components to execute the complete pipeline.
    """
    
    def __init__(self, data_path: str, output_dir: str = 'decision_tree_results'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.setup_logging()
        
        # Components (Dependency Injection - SOLID)
        self.preprocessor = DataPreprocessor(data_path)
        self.trainer = ModelTrainer()
        self.evaluator = None
        self.visualizer = Visualizer(self.output_dir / 'visualizations')
        self.reporter = ReportGenerator(self.output_dir / 'reports')
        
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Setup output directory structure."""
        (self.output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'data').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'logs' / f'analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def create_all_visualizations(self, dt_model, rf_model):
        """Generate all visualizations."""
        self.logger.info("Creating visualizations")
        
        # Decision tree structure
        self.visualizer.plot_decision_tree(dt_model, self.trainer.feature_names, max_depth=3)
        self.visualizer.plot_decision_tree(dt_model, self.trainer.feature_names, max_depth=5)
        
        # Feature importance
        dt_importance = self.visualizer.plot_feature_importance(
            dt_model, self.trainer.feature_names, 'Decision Tree'
        )
        rf_importance = self.visualizer.plot_feature_importance(
            rf_model, self.trainer.feature_names, 'Random Forest'
        )
        
        # Confusion matrices
        for model_name, metrics in self.evaluator.results.items():
            self.visualizer.plot_confusion_matrix(metrics['confusion_matrix'], model_name)
        
        # ROC and PR curves
        self.visualizer.plot_roc_curves(self.evaluator.results, self.trainer.y_test)
        self.visualizer.plot_precision_recall_curves(self.evaluator.results, self.trainer.y_test)
        
        # Comparison dashboard
        comparison_df = self.evaluator.compare_models()
        self.visualizer.create_comparison_dashboard(comparison_df)
        
        return dt_importance, rf_importance
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline."""
        print("=" * 80)
        print("DECISION TREE & RANDOM FOREST ANALYSIS - SPEED DATING MATCH PREDICTION")
        print("=" * 80)
        
        try:
            # Step 1: Load and prepare data
            print("\n[Step 1/9] Loading and preparing data...")
            X, y = self.preprocessor.prepare_dataset()
            
            # Step 2: Split data
            print("[Step 2/9] Splitting data into train/test sets...")
            self.trainer.split_data(X, y, test_size=0.2)
            
            # Step 3: Handle class imbalance
            print("[Step 3/9] Handling class imbalance with SMOTE...")
            self.trainer.handle_imbalance(method='smote')
            
            # Step 4: Train Decision Tree
            print("[Step 4/9] Training and tuning Decision Tree...")
            dt_model, dt_params = self.trainer.tune_decision_tree()
            
            # Step 5: Train Random Forest
            print("[Step 5/9] Training and tuning Random Forest...")
            rf_model = self.trainer.train_random_forest(tune=True)
            
            # Step 6: Evaluate models
            print("[Step 6/9] Evaluating models...")
            self.evaluator = ModelEvaluator(
                self.trainer.X_test, 
                self.trainer.y_test
            )
            dt_metrics = self.evaluator.evaluate_model(dt_model, 'Decision Tree')
            rf_metrics = self.evaluator.evaluate_model(rf_model, 'Random Forest')
            
            # Step 7: Generate visualizations
            print("[Step 7/9] Creating visualizations...")
            dt_importance, rf_importance = self.create_all_visualizations(dt_model, rf_model)
            
            # Step 8: Export results
            print("[Step 8/9] Exporting results...")
            comparison_df = self.evaluator.compare_models()
            
            self.reporter.export_results(
                {'Decision Tree': dt_model, 'Random Forest': rf_model},
                {'Decision Tree': dt_metrics, 'Random Forest': rf_metrics},
                comparison_df,
                self.trainer.feature_names
            )
            
            self.reporter.save_models({
                'Decision Tree': dt_model,
                'Random Forest': rf_model
            })
            
            self.reporter.export_decision_rules(dt_model, self.trainer.feature_names)
            
            # Step 9: Generate report
            print("[Step 9/9] Generating analysis report...")
            self.reporter.generate_markdown_report(
                self.evaluator.results,
                comparison_df,
                dt_params,
                dt_importance,
                rf_importance,
                self.trainer.feature_names
            )
            
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"Results saved to: {self.output_dir.absolute()}")
            print("=" * 80)
            
            # Print summary
            print("\n" + "=" * 80)
            print("PERFORMANCE SUMMARY")
            print("=" * 80)
            print(comparison_df.to_string(index=False))
            print("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            print(f"\n❌ Error: {str(e)}")
            raise


def main():
    """Main entry point."""
    # Configuration
    DATA_PATH = 'Speed_Dating_Data_Cleaned.csv'
    OUTPUT_DIR = 'decision_tree_results'
    
    # Create analyzer and run
    analyzer = DecisionTreeAnalyzer(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
