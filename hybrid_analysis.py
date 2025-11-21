"""
Hybrid Analysis: Apriori-Decision Tree Integration for Speed Dating Analysis

This module combines Association Rule Mining (Apriori) with Decision Tree/Random Forest
insights to validate and interpret tree-based model decisions using association rules,
creating a comprehensive understanding of match prediction patterns.

Author: AIDA_M Final Project
Date: November 2025
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import ast

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
from matplotlib_venn import venn2

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ModelLoader:
    """Loads pre-trained models, Apriori rules, and dataset."""
    
    def __init__(self, data_path: str, models_path: str, apriori_path: str):
        """
        Initialize ModelLoader with file paths.
        
        Args:
            data_path: Path to Speed_Dating_Data_Cleaned.csv
            models_path: Path to decision_tree_results/models directory
            apriori_path: Path to apriori_results/data directory
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_path = data_path
        self.models_path = models_path
        self.apriori_path = apriori_path
        
    def load_models(self) -> Tuple[DecisionTreeClassifier, RandomForestClassifier]:
        """
        Load serialized sklearn models.
        
        Returns:
            Tuple of (decision_tree_model, random_forest_model)
        """
        self.logger.info("Loading machine learning models...")
        
        dt_path = os.path.join(self.models_path, "decision_tree_model.pkl")
        rf_path = os.path.join(self.models_path, "random_forest_model.pkl")
        
        if not os.path.exists(dt_path):
            raise FileNotFoundError(f"Decision Tree model not found: {dt_path}")
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"Random Forest model not found: {rf_path}")
        
        dt_model = joblib.load(dt_path)
        rf_model = joblib.load(rf_path)
        
        # Validate models
        if not isinstance(dt_model, DecisionTreeClassifier):
            raise ValueError("Loaded Decision Tree model is not a DecisionTreeClassifier")
        if not isinstance(rf_model, RandomForestClassifier):
            raise ValueError("Loaded Random Forest model is not a RandomForestClassifier")
        
        self.logger.info(f"✓ Decision Tree loaded (max_depth={dt_model.max_depth})")
        self.logger.info(f"✓ Random Forest loaded (n_estimators={rf_model.n_estimators})")
        
        return dt_model, rf_model
    
    def load_apriori_rules(self) -> pd.DataFrame:
        """
        Load association rules from Apriori analysis.
        
        Returns:
            DataFrame with association rules
        """
        self.logger.info("Loading Apriori association rules...")
        
        # Try to load the smaller match prediction rules file first
        match_rules_path = os.path.join(self.apriori_path, "match_prediction_rules.csv")
        top_rules_path = os.path.join(self.apriori_path, "top_rules_by_lift.csv")
        
        if os.path.exists(match_rules_path):
            df = pd.read_csv(match_rules_path)
            self.logger.info(f"✓ Loaded {len(df)} match prediction rules")
        elif os.path.exists(top_rules_path):
            df = pd.read_csv(top_rules_path)
            self.logger.info(f"✓ Loaded {len(df)} top rules by lift")
        else:
            raise FileNotFoundError("No Apriori rules files found")
        
        # Parse frozenset strings to actual sets
        df['antecedents'] = df['antecedent_str'].apply(self._parse_itemset)
        df['consequents'] = df['consequent_str'].apply(self._parse_itemset)
        
        return df
    
    def _parse_itemset(self, itemset_str: str) -> set:
        """Parse itemset string to set."""
        if pd.isna(itemset_str):
            return set()
        # Remove extra spaces and split by comma
        items = [item.strip() for item in itemset_str.split(',')]
        return set(items)
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load cleaned speed dating dataset.
        
        Returns:
            DataFrame with speed dating data
        """
        self.logger.info("Loading dataset...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        self.logger.info(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def load_all(self) -> Tuple[DecisionTreeClassifier, RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
        """
        Load all required data.
        
        Returns:
            Tuple of (dt_model, rf_model, apriori_rules, dataset)
        """
        dt_model, rf_model = self.load_models()
        apriori_rules = self.load_apriori_rules()
        dataset = self.load_dataset()
        
        return dt_model, rf_model, apriori_rules, dataset


class RuleExtractor:
    """Extracts interpretable rules from Decision Tree models."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_tree_rules(self, tree: DecisionTreeClassifier, 
                          feature_names: List[str],
                          class_names: List[str] = ['No Match', 'Match'],
                          min_samples_pct: float = 0.03) -> List[Dict]:
        """
        Extract all decision paths from trained tree.
        
        Args:
            tree: Trained DecisionTreeClassifier
            feature_names: List of feature names
            class_names: List of class labels
            min_samples_pct: Minimum percentage of samples for a rule to be included
            
        Returns:
            List of rule dictionaries
        """
        self.logger.info("Extracting decision tree rules...")
        
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        
        rules = []
        rule_id = 0
        
        def recurse(node, conditions, depth=0):
            nonlocal rule_id
            
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # Left child (<=)
                left_conditions = conditions + [f"{name} <= {threshold:.3f}"]
                recurse(tree_.children_left[node], left_conditions, depth + 1)
                
                # Right child (>)
                right_conditions = conditions + [f"{name} > {threshold:.3f}"]
                recurse(tree_.children_right[node], right_conditions, depth + 1)
            else:
                # Leaf node
                samples = tree_.n_node_samples[node]
                value = tree_.value[node][0]
                total_samples = tree_.n_node_samples[0]
                
                # Calculate support and confidence
                support = samples / total_samples
                class_idx = np.argmax(value)
                confidence = value[class_idx] / samples if samples > 0 else 0
                
                # Only include rules with sufficient support
                if support >= min_samples_pct:
                    rule = {
                        'rule_id': rule_id,
                        'conditions': conditions.copy(),
                        'prediction': class_names[class_idx],
                        'confidence': float(confidence),
                        'support': float(support),
                        'samples': int(samples),
                        'depth': depth
                    }
                    rules.append(rule)
                    rule_id += 1
        
        recurse(0, [])
        
        self.logger.info(f"✓ Extracted {len(rules)} decision rules (min_support={min_samples_pct})")
        
        return rules
    
    def get_feature_splits(self, tree: DecisionTreeClassifier,
                          feature_names: List[str]) -> pd.DataFrame:
        """
        Get all feature split points used in tree.
        
        Args:
            tree: Trained DecisionTreeClassifier
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature splits
        """
        tree_ = tree.tree_
        
        splits = []
        for node in range(tree_.node_count):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                splits.append({
                    'node_id': node,
                    'feature': feature_names[tree_.feature[node]],
                    'threshold': tree_.threshold[node],
                    'samples_left': tree_.n_node_samples[tree_.children_left[node]],
                    'samples_right': tree_.n_node_samples[tree_.children_right[node]]
                })
        
        return pd.DataFrame(splits)


class RuleMapper:
    """Maps Decision Tree continuous splits to Apriori categorical itemsets."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define threshold mappings for different features
        self.threshold_maps = {
            # Rating features (1-10 scale)
            'attr': {'low': 4.0, 'high': 7.0},
            'sinc': {'low': 4.0, 'high': 7.0},
            'intel': {'low': 4.0, 'high': 7.0},
            'fun': {'low': 4.0, 'high': 7.0},
            'amb': {'low': 4.0, 'high': 7.0},
            'shar': {'low': 4.0, 'high': 7.0},
            'attr_o': {'low': 4.0, 'high': 7.0},
            'sinc_o': {'low': 4.0, 'high': 7.0},
            'intel_o': {'low': 4.0, 'high': 7.0},
            'fun_o': {'low': 4.0, 'high': 7.0},
            'amb_o': {'low': 4.0, 'high': 7.0},
            'shar_o': {'low': 4.0, 'high': 7.0},
            # Age
            'age': {'low': 25.0, 'high': 30.0},
            'age_o': {'low': 25.0, 'high': 30.0},
            # Preference scores
            'preference_match_score': {'low': 60.0, 'high': 80.0},
            # Average ratings
            'avg_rating_given': {'low': 4.0, 'high': 7.0},
            'avg_rating_received': {'low': 4.0, 'high': 7.0},
            # Differences
            'attr_diff': {'low': -2.0, 'high': 2.0},
            'sinc_diff': {'low': -2.0, 'high': 2.0},
            'intel_diff': {'low': -2.0, 'high': 2.0},
            'fun_diff': {'low': -2.0, 'high': 2.0},
            'amb_diff': {'low': -2.0, 'high': 2.0},
            'shar_diff': {'low': -2.0, 'high': 2.0},
        }
    
    def categorize_threshold(self, feature: str, threshold: float, direction: str) -> str:
        """
        Map continuous split to Apriori category.
        
        Args:
            feature: Feature name
            threshold: Split threshold
            direction: '<=' or '>'
            
        Returns:
            Apriori itemset string or None if not mappable
        """
        # Extract base feature name (without _o, _diff, or other suffixes)
        base_feature = feature
        for suffix in ['_diff', '_o', '_1', '_2', '_3']:
            base_feature = base_feature.replace(suffix, '')
        
        # Check if feature has defined thresholds
        # Try exact match first, then base feature
        if feature in self.threshold_maps:
            thresholds = self.threshold_maps[feature]
        elif base_feature in self.threshold_maps:
            thresholds = self.threshold_maps[base_feature]
        else:
            return None
        
        # Determine category based on threshold and direction
        if direction == '<=':
            if threshold <= thresholds['low']:
                category = 'Low'
            elif threshold <= thresholds['high']:
                category = 'Medium'
            else:
                category = 'High'  # Changed to allow high category
        else:  # '>'
            if threshold >= thresholds['high']:
                category = 'High'
            elif threshold >= thresholds['low']:
                category = 'Medium'  # or High
            else:
                category = 'Low'
        
        # Format itemset based on feature type
        if feature.endswith('_o'):
            # Partner rating
            itemset = f"{feature}_cat_{category}_Rcvd"
        elif feature == 'age':
            # Age categories
            age_map = {'Low': 'Young', 'Medium': 'Middle', 'High': 'Mature'}
            itemset = f"age_cat_{age_map.get(category, category)}"
        elif '_diff' in feature:
            # Difference features - use simpler naming
            base = feature.replace('_diff', '')
            itemset = f"{base}_diff_cat_{category}"
        else:
            # Self rating or other features
            itemset = f"{feature}_cat_{category}"
        
        return itemset
    
    def map_tree_rule_to_apriori(self, tree_rule: Dict) -> Dict:
        """
        Convert Decision Tree rule to Apriori-compatible format.
        
        Args:
            tree_rule: Decision tree rule dictionary
            
        Returns:
            Dictionary with mapped itemsets
        """
        mapped_items = set()
        unmapped_conditions = []
        
        for condition in tree_rule['conditions']:
            # Parse condition
            parts = condition.split()
            if len(parts) != 3:
                unmapped_conditions.append(condition)
                continue
            
            feature = parts[0]
            operator = parts[1]
            threshold = float(parts[2])
            
            # Map to category
            itemset = self.categorize_threshold(feature, threshold, operator)
            
            if itemset:
                mapped_items.add(itemset)
            else:
                unmapped_conditions.append(condition)
        
        # Add prediction as consequent
        if tree_rule['prediction'] == 'Match':
            consequent = {'match_outcome_Match'}
        else:
            consequent = {'match_outcome_No_Match'}
        
        # Calculate mapping confidence
        total_conditions = len(tree_rule['conditions'])
        mapped_conditions = total_conditions - len(unmapped_conditions)
        mapping_confidence = mapped_conditions / total_conditions if total_conditions > 0 else 0
        
        return {
            'rule_id': tree_rule['rule_id'],
            'antecedent_items': mapped_items,
            'consequent_items': consequent,
            'original_conditions': tree_rule['conditions'],
            'unmapped_conditions': unmapped_conditions,
            'mapping_confidence': mapping_confidence,
            'dt_confidence': tree_rule['confidence'],
            'dt_support': tree_rule['support'],
            'dt_samples': tree_rule['samples'],
            'prediction': tree_rule['prediction']
        }


class RuleValidator:
    """Validates and compares Decision Tree rules against Apriori rules."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def find_matching_apriori_rules(self, mapped_tree_rule: Dict,
                                     apriori_rules: pd.DataFrame,
                                     min_overlap: int = 1) -> pd.DataFrame:
        """
        Search for Apriori rules matching tree rule.
        
        Args:
            mapped_tree_rule: Mapped tree rule dictionary
            apriori_rules: DataFrame of Apriori rules
            min_overlap: Minimum number of overlapping items
            
        Returns:
            DataFrame of matching Apriori rules with similarity scores
        """
        tree_antecedent = mapped_tree_rule['antecedent_items']
        tree_consequent = mapped_tree_rule['consequent_items']
        
        if len(tree_antecedent) == 0:
            return pd.DataFrame()
        
        matches = []
        
        for idx, apriori_rule in apriori_rules.iterrows():
            apriori_ant = apriori_rule['antecedents']
            apriori_cons = apriori_rule['consequents']
            
            # Calculate overlap
            ant_overlap = len(tree_antecedent & apriori_ant)
            cons_match = len(tree_consequent & apriori_cons) > 0
            
            if ant_overlap >= min_overlap and cons_match:
                # Calculate similarity score
                ant_jaccard = ant_overlap / len(tree_antecedent | apriori_ant)
                
                match = apriori_rule.to_dict()
                match['antecedent_overlap'] = ant_overlap
                match['antecedent_similarity'] = ant_jaccard
                match['consequent_match'] = cons_match
                
                matches.append(match)
        
        if matches:
            matches_df = pd.DataFrame(matches)
            matches_df = matches_df.sort_values('lift', ascending=False)
            return matches_df
        
        return pd.DataFrame()
    
    def validate_rule(self, tree_rule: Dict, mapped_rule: Dict,
                     apriori_matches: pd.DataFrame) -> Dict:
        """
        Compare metrics between Decision Tree rule and Apriori rules.
        
        Args:
            tree_rule: Original tree rule
            mapped_rule: Mapped tree rule
            apriori_matches: DataFrame of matching Apriori rules
            
        Returns:
            Validation result dictionary
        """
        if len(apriori_matches) == 0:
            return {
                'rule_id': tree_rule['rule_id'],
                'validation_status': 'NO_MATCH',
                'dt_confidence': tree_rule['confidence'],
                'dt_support': tree_rule['support'],
                'apriori_confidence': None,
                'apriori_lift': None,
                'confidence_delta': None,
                'agreement_score': 0.0,
                'matching_rules_count': 0,
                'best_match': None
            }
        
        # Get best matching rule (highest lift)
        best_match = apriori_matches.iloc[0]
        
        # Calculate metrics
        conf_delta = abs(tree_rule['confidence'] - best_match['confidence'])
        
        # Calculate agreement score (0-100)
        agreement_score = self.calculate_agreement_score(
            tree_rule, mapped_rule, best_match
        )
        
        # Determine validation status
        if agreement_score >= 80:
            status = 'CONFIRMED'
        elif agreement_score >= 50:
            status = 'PARTIAL'
        elif agreement_score >= 20:
            status = 'CONFLICTING'
        else:
            status = 'WEAK'
        
        return {
            'rule_id': tree_rule['rule_id'],
            'validation_status': status,
            'dt_confidence': tree_rule['confidence'],
            'dt_support': tree_rule['support'],
            'apriori_confidence': best_match['confidence'],
            'apriori_lift': best_match['lift'],
            'apriori_support': best_match['support'],
            'confidence_delta': conf_delta,
            'agreement_score': agreement_score,
            'matching_rules_count': len(apriori_matches),
            'best_match': best_match.to_dict(),
            'antecedent_similarity': best_match.get('antecedent_similarity', 0),
            'mapping_confidence': mapped_rule['mapping_confidence']
        }
    
    def calculate_agreement_score(self, tree_rule: Dict, mapped_rule: Dict,
                                  apriori_rule: pd.Series) -> float:
        """
        Compute overall agreement score between rules.
        
        Args:
            tree_rule: Original tree rule
            mapped_rule: Mapped tree rule
            apriori_rule: Apriori rule Series
            
        Returns:
            Score from 0-100
        """
        # Factor weights
        ITEMSET_WEIGHT = 0.40
        CONFIDENCE_WEIGHT = 0.30
        SUPPORT_WEIGHT = 0.15
        LIFT_WEIGHT = 0.15
        
        # 1. Itemset overlap (Jaccard similarity)
        itemset_score = apriori_rule.get('antecedent_similarity', 0) * 100
        
        # 2. Confidence similarity (1 - normalized difference)
        conf_diff = abs(tree_rule['confidence'] - apriori_rule['confidence'])
        confidence_score = max(0, (1 - conf_diff) * 100)
        
        # 3. Support correlation (normalized)
        support_ratio = min(tree_rule['support'], apriori_rule['support']) / \
                       max(tree_rule['support'], apriori_rule['support'])
        support_score = support_ratio * 100
        
        # 4. Lift strength (normalized to 0-100, lift > 3 = 100)
        lift_score = min(100, (apriori_rule['lift'] / 3.0) * 100)
        
        # Combined score
        total_score = (
            itemset_score * ITEMSET_WEIGHT +
            confidence_score * CONFIDENCE_WEIGHT +
            support_score * SUPPORT_WEIGHT +
            lift_score * LIFT_WEIGHT
        )
        
        # Apply penalty for low mapping confidence
        total_score *= mapped_rule['mapping_confidence']
        
        return total_score


class InsightGenerator:
    """Generates insights from validated rules."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def identify_strongest_patterns(self, validated_rules: List[Dict],
                                    top_n: int = 20) -> pd.DataFrame:
        """
        Rank patterns by combined evidence strength.
        
        Args:
            validated_rules: List of validation results
            top_n: Number of top patterns to return
            
        Returns:
            DataFrame of strongest patterns
        """
        self.logger.info("Identifying strongest patterns...")
        
        # Filter for confirmed and partial matches
        strong_rules = [
            r for r in validated_rules
            if r['validation_status'] in ['CONFIRMED', 'PARTIAL']
        ]
        
        if not strong_rules:
            self.logger.warning("No strong patterns found!")
            return pd.DataFrame()
        
        # Calculate pattern strength
        for rule in strong_rules:
            pattern_strength = (
                rule['dt_confidence'] * 0.3 +
                (rule['apriori_confidence'] or 0) * 0.3 +
                (rule['apriori_lift'] or 0) * 0.1 +
                rule['agreement_score'] * 0.003  # Scale to similar range
            )
            rule['pattern_strength'] = pattern_strength
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(strong_rules)
        df = df.sort_values('pattern_strength', ascending=False).head(top_n)
        
        self.logger.info(f"✓ Identified {len(df)} strongest patterns")
        
        return df
    
    def find_novel_tree_patterns(self, validated_rules: List[Dict]) -> List[Dict]:
        """
        Identify Decision Tree rules with no Apriori support.
        
        Args:
            validated_rules: List of validation results
            
        Returns:
            List of novel patterns
        """
        self.logger.info("Finding novel tree patterns...")
        
        novel = [
            r for r in validated_rules
            if r['validation_status'] == 'NO_MATCH' and r['dt_confidence'] > 0.7
        ]
        
        self.logger.info(f"✓ Found {len(novel)} novel patterns")
        
        return novel
    
    def analyze_contradictions(self, validated_rules: List[Dict]) -> pd.DataFrame:
        """
        Analyze rules where tree and Apriori disagree.
        
        Args:
            validated_rules: List of validation results
            
        Returns:
            DataFrame with contradiction analysis
        """
        self.logger.info("Analyzing contradictions...")
        
        conflicts = [
            r for r in validated_rules
            if r['validation_status'] == 'CONFLICTING'
        ]
        
        if not conflicts:
            self.logger.info("No contradictions found")
            return pd.DataFrame()
        
        df = pd.DataFrame(conflicts)
        self.logger.info(f"✓ Analyzed {len(df)} contradictions")
        
        return df


class Visualizer:
    """Creates visualizations for hybrid analysis."""
    
    def __init__(self, output_dir: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_confidence_comparison(self, validated_rules: List[Dict]) -> None:
        """Create scatter plot comparing DT and Apriori confidence."""
        self.logger.info("Creating confidence comparison plot...")
        
        # Filter rules with Apriori matches
        data = [
            r for r in validated_rules
            if r['apriori_confidence'] is not None
        ]
        
        if not data:
            self.logger.warning("No data for confidence comparison")
            return
        
        df = pd.DataFrame(data)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by validation status
        colors = {
            'CONFIRMED': 'green',
            'PARTIAL': 'orange',
            'CONFLICTING': 'red',
            'WEAK': 'gray'
        }
        
        for status, color in colors.items():
            mask = df['validation_status'] == status
            if mask.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=df[mask]['dt_confidence'],
                    y=df[mask]['apriori_confidence'],
                    mode='markers',
                    name=status,
                    marker=dict(
                        size=df[mask]['dt_support'] * 500,
                        color=color,
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Rule {r}" for r in df[mask]['rule_id']],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'DT Conf: %{x:.3f}<br>' +
                                  'Apriori Conf: %{y:.3f}<br>' +
                                  '<extra></extra>'
                ))
        
        # Add diagonal line (perfect agreement)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Agreement',
            line=dict(color='black', dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title='Confidence Comparison: Decision Tree vs Apriori',
            xaxis_title='Decision Tree Confidence',
            yaxis_title='Apriori Confidence',
            hovermode='closest',
            width=900,
            height=700
        )
        
        output_path = os.path.join(self.output_dir, 'confidence_comparison.html')
        fig.write_html(output_path)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def create_validation_summary_chart(self, validated_rules: List[Dict]) -> None:
        """Create summary chart of validation statuses."""
        self.logger.info("Creating validation summary chart...")
        
        df = pd.DataFrame(validated_rules)
        
        # Count by status
        status_counts = df['validation_status'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=status_counts.index,
                y=status_counts.values,
                text=status_counts.values,
                textposition='auto',
                marker=dict(
                    color=['green', 'orange', 'gray', 'red'],
                    line=dict(color='white', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title='Rule Validation Status Distribution',
            xaxis_title='Validation Status',
            yaxis_title='Number of Rules',
            showlegend=False,
            width=800,
            height=500
        )
        
        output_path = os.path.join(self.output_dir, 'validation_summary.html')
        fig.write_html(output_path)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def create_agreement_score_distribution(self, validated_rules: List[Dict]) -> None:
        """Create histogram of agreement scores."""
        self.logger.info("Creating agreement score distribution...")
        
        df = pd.DataFrame(validated_rules)
        
        fig = go.Figure(data=[
            go.Histogram(
                x=df['agreement_score'],
                nbinsx=20,
                marker=dict(
                    color=df['agreement_score'],
                    colorscale='RdYlGn',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Distribution of Agreement Scores',
            xaxis_title='Agreement Score (0-100)',
            yaxis_title='Number of Rules',
            width=900,
            height=500
        )
        
        output_path = os.path.join(self.output_dir, 'agreement_score_distribution.html')
        fig.write_html(output_path)
        self.logger.info(f"✓ Saved: {output_path}")
    
    def create_pattern_strength_chart(self, strongest_patterns: pd.DataFrame) -> None:
        """Create bar chart of strongest patterns."""
        if strongest_patterns.empty:
            return
        
        self.logger.info("Creating pattern strength chart...")
        
        top_10 = strongest_patterns.head(10).copy()
        top_10['rule_label'] = top_10['rule_id'].apply(lambda x: f"Rule {x}")
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_10['pattern_strength'],
                y=top_10['rule_label'],
                orientation='h',
                text=top_10['pattern_strength'].round(3),
                textposition='auto',
                marker=dict(
                    color=top_10['agreement_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Agreement<br>Score')
                )
            )
        ])
        
        fig.update_layout(
            title='Top 10 Strongest Validated Patterns',
            xaxis_title='Pattern Strength Score',
            yaxis_title='Rule',
            height=600,
            width=900
        )
        
        output_path = os.path.join(self.output_dir, 'pattern_strength.html')
        fig.write_html(output_path)
        self.logger.info(f"✓ Saved: {output_path}")


class ReportGenerator:
    """Generates reports and exports results."""
    
    def __init__(self, output_dir: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_to_csv(self, validated_rules: List[Dict], tree_rules: List[Dict],
                     strongest: pd.DataFrame, novel: List[Dict],
                     contradictions: pd.DataFrame) -> None:
        """Export all results to CSV files."""
        self.logger.info("Exporting results to CSV...")
        
        data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # 1. All validated patterns
        df_validated = pd.DataFrame(validated_rules)
        validated_path = os.path.join(data_dir, 'validated_patterns.csv')
        df_validated.to_csv(validated_path, index=False)
        self.logger.info(f"✓ Exported: {validated_path}")
        
        # 2. Decision tree rules
        df_tree = pd.DataFrame(tree_rules)
        df_tree['conditions_str'] = df_tree['conditions'].apply(lambda x: ' AND '.join(x))
        tree_path = os.path.join(data_dir, 'decision_tree_rules.csv')
        df_tree.to_csv(tree_path, index=False)
        self.logger.info(f"✓ Exported: {tree_path}")
        
        # 3. Strongest patterns
        if not strongest.empty:
            strongest_path = os.path.join(data_dir, 'strongest_patterns.csv')
            strongest.to_csv(strongest_path, index=False)
            self.logger.info(f"✓ Exported: {strongest_path}")
        
        # 4. Novel patterns
        if novel:
            df_novel = pd.DataFrame(novel)
            novel_path = os.path.join(data_dir, 'novel_patterns.csv')
            df_novel.to_csv(novel_path, index=False)
            self.logger.info(f"✓ Exported: {novel_path}")
        
        # 5. Contradictions
        if not contradictions.empty:
            contra_path = os.path.join(data_dir, 'contradictions.csv')
            contradictions.to_csv(contra_path, index=False)
            self.logger.info(f"✓ Exported: {contra_path}")
    
    def generate_markdown_report(self, validated_rules: List[Dict],
                                 tree_rules: List[Dict],
                                 strongest: pd.DataFrame,
                                 novel: List[Dict]) -> None:
        """Generate comprehensive Markdown report."""
        self.logger.info("Generating Markdown report...")
        
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Calculate statistics
        total_tree_rules = len(tree_rules)
        total_validated = len(validated_rules)
        
        status_counts = pd.DataFrame(validated_rules)['validation_status'].value_counts()
        confirmed = status_counts.get('CONFIRMED', 0)
        partial = status_counts.get('PARTIAL', 0)
        no_match = status_counts.get('NO_MATCH', 0)
        
        confirmation_rate = (confirmed / total_validated * 100) if total_validated > 0 else 0
        
        # Generate report content
        report = f"""# Hybrid Analysis Report: Apriori + Decision Tree Integration

## Executive Summary

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Overview Statistics

- **Total Decision Tree Rules Analyzed:** {total_tree_rules}
- **Rules with Sufficient Support:** {total_validated}
- **Validated Patterns (Confirmed/Partial):** {confirmed + partial}
- **Strong Confirmation Rate:** {confirmation_rate:.1f}%
- **Novel Tree Insights (No Apriori Match):** {no_match}

### Validation Status Distribution

| Status | Count | Percentage |
|--------|-------|------------|
"""
        
        for status, count in status_counts.items():
            pct = count / total_validated * 100
            report += f"| {status} | {count} | {pct:.1f}% |\n"
        
        report += f"""

---

## Key Findings

### 1. Strongest Validated Patterns

These patterns show strong agreement between Decision Tree splits and Apriori association rules:

"""
        
        if not strongest.empty:
            for idx, row in strongest.head(10).iterrows():
                report += f"""
**Pattern #{row['rule_id']}**
- **Validation Status:** {row['validation_status']}
- **Agreement Score:** {row['agreement_score']:.1f}/100
- **Decision Tree Confidence:** {row['dt_confidence']:.3f} ({row['dt_confidence']*100:.1f}%)
- **Apriori Confidence:** {row.get('apriori_confidence', 0):.3f}
- **Apriori Lift:** {row.get('apriori_lift', 0):.2f}
- **Pattern Strength:** {row.get('pattern_strength', 0):.3f}

"""
        
        report += f"""
### 2. Novel Tree Insights

Patterns discovered by Decision Tree but not strongly supported by Apriori rules:

- **Total Novel Patterns:** {len(novel)}
- **High-Confidence Novel Rules:** {len([n for n in novel if n['dt_confidence'] > 0.8])}

These patterns may indicate:
- Nuanced interactions not captured by Apriori's minimum support threshold
- Complex feature combinations
- Continuous threshold effects not reflected in categorical Apriori itemsets

"""
        
        if novel:
            report += "\n**Top Novel Patterns:**\n\n"
            for rule in novel[:5]:
                report += f"- Rule {rule['rule_id']}: Confidence {rule['dt_confidence']:.3f}, Support {rule['dt_support']:.3f}\n"
        
        report += f"""

### 3. Method Agreement Analysis

**Strong Agreement (CONFIRMED):**
- Patterns where both methods independently identified the same relationships
- High confidence similarity and strong lift values
- Most reliable predictors for match outcomes

**Partial Agreement (PARTIAL):**
- Some overlap in identified patterns
- May differ in confidence levels or subset of conditions
- Still provide useful validation

**No Match:**
- Decision Tree patterns with no corresponding Apriori rules
- May represent overfitting or unique tree discoveries
- Require further investigation

---

## Interpretation Guidelines

### Agreement Score Interpretation

The agreement score (0-100) combines multiple factors:

- **Itemset Overlap (40% weight):** How well tree conditions map to Apriori items
- **Confidence Similarity (30% weight):** Agreement in prediction confidence
- **Support Correlation (15% weight):** Similar prevalence in dataset
- **Lift Strength (15% weight):** Strength of association in Apriori

**Score Ranges:**
- **80-100:** Strong confirmation - Both methods agree
- **50-79:** Partial support - Some agreement with differences
- **20-49:** Conflicting - Methods disagree
- **0-19:** Weak/No match - No corresponding Apriori rule

### Feature Insights

The most important features from Random Forest analysis should align with
features appearing frequently in high-lift Apriori rules. Key features include:

- **attr/attr_o:** Attractiveness ratings (given/received)
- **fun/fun_o:** Fun ratings (given/received)
- **shar/shar_o:** Shared interests ratings (given/received)
- **sinc/sinc_o:** Sincerity ratings (given/received)
- **intel/intel_o:** Intelligence ratings (given/received)

---

## Recommendations

### For Match Prediction

1. **Prioritize CONFIRMED patterns** for most reliable predictions
2. **Investigate PARTIAL patterns** for additional insights
3. **Use ensemble approach** combining both tree and association rule strengths
4. **Monitor NOVEL patterns** for potential overfitting

### For Future Analysis

1. **Refine threshold mappings** between continuous and categorical features
2. **Experiment with different Apriori support/confidence thresholds**
3. **Consider temporal patterns** if date progression data is available
4. **Explore additional ensemble methods** (XGBoost, neural networks)

---

## Methodology

### Analysis Pipeline

1. **Model Loading:** Loaded pre-trained Decision Tree and Random Forest models
2. **Rule Extraction:** Extracted decision paths from tree model
3. **Rule Mapping:** Mapped continuous splits to categorical Apriori itemsets
4. **Validation:** Compared tree rules against association rules
5. **Scoring:** Calculated agreement scores based on multiple factors
6. **Insights:** Identified strongest patterns and contradictions

### Threshold Mappings

| Feature Type | Low | Medium | High |
|--------------|-----|--------|------|
| Rating (1-10) | ≤4 | 4-7 | >7 |
| Age | <25 | 25-30 | >30 |

---

*Report generated by Hybrid Analysis System*
*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_path = os.path.join(reports_dir, 'hybrid_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"✓ Report saved: {report_path}")


class HybridAnalyzer:
    """Main orchestrator for hybrid analysis."""
    
    def __init__(self, data_path: str, models_path: str, apriori_path: str,
                 output_path: str = "hybrid_results"):
        """
        Initialize HybridAnalyzer.
        
        Args:
            data_path: Path to cleaned dataset
            models_path: Path to models directory
            apriori_path: Path to Apriori results directory
            output_path: Output directory for results
        """
        self.logger = self._setup_logging()
        self.output_path = output_path
        
        # Initialize components
        self.loader = ModelLoader(data_path, models_path, apriori_path)
        self.extractor = RuleExtractor()
        self.mapper = RuleMapper()
        self.validator = RuleValidator()
        self.insight_generator = InsightGenerator()
        self.visualizer = Visualizer(os.path.join(output_path, 'visualizations'))
        self.reporter = ReportGenerator(output_path)
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'reports'), exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('HybridAnalyzer')
        return logger
    
    def run_analysis(self) -> None:
        """Execute complete hybrid analysis pipeline."""
        
        print("\n" + "="*80)
        print("HYBRID ANALYSIS: Apriori-Decision Tree Integration")
        print("="*80 + "\n")
        
        # Step 1: Load all data
        print("[Step 1/7] Loading data and models...")
        dt_model, rf_model, apriori_rules, dataset = self.loader.load_all()
        
        # Get feature names from dataset (assuming match column is target)
        feature_cols = [col for col in dataset.columns if col != 'match']
        
        # Step 2: Extract decision tree rules
        print("\n[Step 2/7] Extracting decision tree rules...")
        tree_rules = self.extractor.extract_tree_rules(
            dt_model,
            feature_cols,
            class_names=['No Match', 'Match'],
            min_samples_pct=0.01  # Lower threshold to extract more rules
        )
        
        # Step 3: Map rules to Apriori format
        print("\n[Step 3/7] Mapping tree rules to Apriori format...")
        mapped_rules = []
        for rule in tree_rules:
            mapped = self.mapper.map_tree_rule_to_apriori(rule)
            mapped_rules.append(mapped)
        
        mapped_count = sum(1 for r in mapped_rules if r['mapping_confidence'] > 0.5)
        self.logger.info(f"✓ Successfully mapped {mapped_count}/{len(mapped_rules)} rules (>50% confidence)")
        
        # Step 4: Validate rules
        print("\n[Step 4/7] Validating rules against Apriori...")
        validated_results = []
        
        for tree_rule, mapped_rule in zip(tree_rules, mapped_rules):
            # Find matching Apriori rules
            matches = self.validator.find_matching_apriori_rules(
                mapped_rule, apriori_rules, min_overlap=1
            )
            
            # Validate
            validation = self.validator.validate_rule(tree_rule, mapped_rule, matches)
            validated_results.append(validation)
        
        status_summary = pd.Series([r['validation_status'] for r in validated_results]).value_counts()
        self.logger.info(f"✓ Validation complete:\n{status_summary}")
        
        # Step 5: Generate insights
        print("\n[Step 5/7] Generating insights...")
        strongest = self.insight_generator.identify_strongest_patterns(validated_results, top_n=20)
        novel = self.insight_generator.find_novel_tree_patterns(validated_results)
        contradictions = self.insight_generator.analyze_contradictions(validated_results)
        
        # Step 6: Create visualizations
        print("\n[Step 6/7] Creating visualizations...")
        self.visualizer.plot_confidence_comparison(validated_results)
        self.visualizer.create_validation_summary_chart(validated_results)
        self.visualizer.create_agreement_score_distribution(validated_results)
        self.visualizer.create_pattern_strength_chart(strongest)
        
        # Step 7: Export results and generate report
        print("\n[Step 7/7] Exporting results and generating report...")
        self.reporter.export_to_csv(
            validated_results, tree_rules, strongest, novel, contradictions
        )
        self.reporter.generate_markdown_report(
            validated_results, tree_rules, strongest, novel
        )
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n✓ Results saved to: {self.output_path}/")
        print(f"  - Data exports: {os.path.join(self.output_path, 'data')}/")
        print(f"  - Visualizations: {os.path.join(self.output_path, 'visualizations')}/")
        print(f"  - Reports: {os.path.join(self.output_path, 'reports')}/")
        
        # Print key statistics
        print(f"\n{'='*80}")
        print("KEY STATISTICS")
        print(f"{'='*80}")
        print(f"Total Decision Tree Rules: {len(tree_rules)}")
        print(f"Successfully Mapped Rules: {mapped_count}")
        print(f"Confirmed Patterns: {status_summary.get('CONFIRMED', 0)}")
        print(f"Partial Matches: {status_summary.get('PARTIAL', 0)}")
        print(f"Novel Patterns: {len(novel)}")
        print(f"Strongest Patterns Identified: {len(strongest)}")
        print("="*80 + "\n")


def main():
    """Main execution function."""
    
    # Configuration
    DATA_PATH = "Speed_Dating_Data_Cleaned.csv"
    MODELS_PATH = "decision_tree_results/models"
    APRIORI_PATH = "apriori_results/data"
    OUTPUT_PATH = "hybrid_results"
    
    # Run analysis
    try:
        analyzer = HybridAnalyzer(DATA_PATH, MODELS_PATH, APRIORI_PATH, OUTPUT_PATH)
        analyzer.run_analysis()
        
        print("\n✓ Hybrid analysis completed successfully!")
        print(f"\nTo view results:")
        print(f"  1. Open {OUTPUT_PATH}/reports/hybrid_analysis_report.md")
        print(f"  2. Explore visualizations in {OUTPUT_PATH}/visualizations/")
        print(f"  3. Analyze data in {OUTPUT_PATH}/data/\n")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        logging.exception("Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
