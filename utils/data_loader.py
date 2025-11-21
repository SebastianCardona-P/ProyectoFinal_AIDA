"""
Data loader utilities for the dating market simulation.

This module handles loading ML models, association rules, and dataset statistics.
"""

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from config.simulation_config import config


# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class ModelLoader:
    """Loader for Random Forest model and related data."""
    
    def __init__(self):
        """Initialize the model loader."""
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        
    def load_random_forest(self) -> Any:
        """
        Load the pre-trained Random Forest model.
        
        Returns:
            Loaded Random Forest model.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if not os.path.exists(config.RANDOM_FOREST_PATH):
            raise FileNotFoundError(f"Random Forest model not found at {config.RANDOM_FOREST_PATH}")
        
        logger.info(f"Loading Random Forest model from {config.RANDOM_FOREST_PATH}")
        self.model = joblib.load(config.RANDOM_FOREST_PATH)
        logger.info("Random Forest model loaded successfully")
        return self.model
    
    def load_feature_importance(self) -> pd.DataFrame:
        """
        Load feature importance data.
        
        Returns:
            DataFrame with feature names and importance values.
        """
        if not os.path.exists(config.FEATURE_IMPORTANCE_PATH):
            logger.warning(f"Feature importance file not found at {config.FEATURE_IMPORTANCE_PATH}")
            return pd.DataFrame()
        
        logger.info(f"Loading feature importance from {config.FEATURE_IMPORTANCE_PATH}")
        self.feature_importance = pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
        self.feature_names = self.feature_importance['feature'].tolist()
        logger.info(f"Loaded {len(self.feature_importance)} features")
        return self.feature_importance
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return.
            
        Returns:
            DataFrame with top N features.
        """
        if self.feature_importance is None:
            self.load_feature_importance()
        
        return self.feature_importance.head(n)


class RulesLoader:
    """Loader for Apriori association rules."""
    
    def __init__(self):
        """Initialize the rules loader."""
        self.rules = None
        
    def load_association_rules(self, max_rows: int = 1000) -> pd.DataFrame:
        """
        Load association rules from CSV.
        
        Args:
            max_rows: Maximum number of rules to load (for performance).
            
        Returns:
            DataFrame with association rules.
        """
        if not os.path.exists(config.ASSOCIATION_RULES_PATH):
            logger.warning(f"Association rules file not found at {config.ASSOCIATION_RULES_PATH}")
            return pd.DataFrame()
        
        logger.info(f"Loading association rules from {config.ASSOCIATION_RULES_PATH}")
        try:
            # Load only top rules by lift
            self.rules = pd.read_csv(config.ASSOCIATION_RULES_PATH, nrows=max_rows)
            
            # Sort by lift descending if lift column exists
            if 'lift' in self.rules.columns:
                self.rules = self.rules.sort_values('lift', ascending=False)
            
            logger.info(f"Loaded {len(self.rules)} association rules")
            return self.rules
        except Exception as e:
            logger.error(f"Error loading association rules: {e}")
            return pd.DataFrame()
    
    def get_top_rules(self, n: int = 10, min_confidence: float = 0.5) -> pd.DataFrame:
        """
        Get top N rules with minimum confidence.
        
        Args:
            n: Number of top rules to return.
            min_confidence: Minimum confidence threshold.
            
        Returns:
            DataFrame with top N rules.
        """
        if self.rules is None or self.rules.empty:
            self.load_association_rules()
        
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()
        
        # Filter by confidence if column exists
        if 'confidence' in self.rules.columns:
            filtered = self.rules[self.rules['confidence'] >= min_confidence]
        else:
            filtered = self.rules
        
        return filtered.head(n)


class DataStatistics:
    """Extract and store dataset statistics for agent generation."""
    
    def __init__(self):
        """Initialize data statistics."""
        self.data = None
        self.stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load cleaned dataset.
        
        Returns:
            DataFrame with cleaned data.
        """
        if not os.path.exists(config.CLEANED_DATA_PATH):
            raise FileNotFoundError(f"Cleaned data not found at {config.CLEANED_DATA_PATH}")
        
        logger.info(f"Loading cleaned data from {config.CLEANED_DATA_PATH}")
        self.data = pd.read_csv(config.CLEANED_DATA_PATH)
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics from the dataset.
        
        Returns:
            Dictionary with statistics for agent generation.
        """
        if self.data is None:
            self.load_data()
        
        logger.info("Computing dataset statistics")
        
        # Age statistics
        self.stats['age'] = {
            'mean': float(self.data['age'].mean()),
            'std': float(self.data['age'].std()),
            'min': float(self.data['age'].min()),
            'max': float(self.data['age'].max())
        }
        
        # Race distribution
        race_cols = [col for col in self.data.columns if col.startswith('race_') and col not in ['race_o']]
        if race_cols:
            race_dist = {}
            for col in race_cols:
                # Extract numeric part and convert to int (e.g., 'race_1.0' -> 1)
                race_num_str = col.split('_')[1]
                try:
                    race_num = int(float(race_num_str))
                    race_dist[race_num] = float(self.data[col].sum() / len(self.data))
                except (ValueError, IndexError):
                    continue
            self.stats['race_distribution'] = race_dist
        
        # Field distribution
        field_cols = [col for col in self.data.columns if col.startswith('field_') and col not in ['field', 'field_cd']]
        if field_cols:
            field_dist = {}
            for col in field_cols:
                # Extract numeric part and convert to int (e.g., 'field_1.0' -> 1)
                field_num_str = col.split('_')[1]
                try:
                    field_num = int(float(field_num_str))
                    field_dist[field_num] = float(self.data[col].sum() / len(self.data))
                except (ValueError, IndexError):
                    continue
            self.stats['field_distribution'] = field_dist
        
        # Rating statistics (for attributes)
        rating_attrs = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
        for attr in rating_attrs:
            if attr in self.data.columns:
                self.stats[attr] = {
                    'mean': float(self.data[attr].mean()),
                    'std': float(self.data[attr].std()),
                    'min': float(self.data[attr].min()),
                    'max': float(self.data[attr].max())
                }
        
        # Gender distribution
        if 'gender' in self.data.columns:
            male_count = self.data['gender'].sum()
            total = len(self.data)
            self.stats['gender'] = {
                'male_ratio': float(male_count / total),
                'female_ratio': float(1 - (male_count / total))
            }
        
        logger.info("Statistics computed successfully")
        return self.stats
    
    def get_stat(self, key: str) -> Any:
        """
        Get a specific statistic.
        
        Args:
            key: Statistic key.
            
        Returns:
            Statistic value or None if not found.
        """
        if not self.stats:
            self.compute_statistics()
        
        return self.stats.get(key)
    
    def sample_from_distribution(self, attribute: str) -> float:
        """
        Sample a value from an attribute's distribution.
        
        Args:
            attribute: Attribute name (e.g., 'attr', 'age').
            
        Returns:
            Sampled value.
        """
        if attribute not in self.stats:
            logger.warning(f"Attribute {attribute} not in statistics")
            return 5.0  # Default mid-range value
        
        stat = self.stats[attribute]
        if 'mean' in stat and 'std' in stat:
            value = np.random.normal(stat['mean'], stat['std'])
            # Clip to min/max if available
            if 'min' in stat and 'max' in stat:
                value = np.clip(value, stat['min'], stat['max'])
            return float(value)
        
        return 5.0  # Default


# Singleton instances for easy access
model_loader = ModelLoader()
rules_loader = RulesLoader()
data_stats = DataStatistics()
