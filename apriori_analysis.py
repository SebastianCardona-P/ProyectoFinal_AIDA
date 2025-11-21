"""
Apriori Association Rule Mining - Speed Dating Analysis
=======================================================

This module implements comprehensive association rule mining using the Apriori algorithm
to discover patterns in speed dating data. It identifies combinations of attributes and
preferences that lead to successful matches.

Author: Data Analysis Team
Date: November 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import traceback

# Apriori-specific imports
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration and Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Apriori thresholds
MIN_SUPPORT = 0.08  # Increased to reduce memory usage
MIN_CONFIDENCE = 0.4
MIN_LIFT = 1.2

# Discretization parameters
N_BINS = 3  # Low, Medium, High
N_BINS_INCOME = 3  # Low, Medium, High income
N_BINS_AGE = 3  # Young, Middle, Mature

# Output configuration
OUTPUT_DIR = 'apriori_results'
DPI = 300

# Color schemes
COLOR_PALETTE = 'viridis'
NETWORK_COLORS = {
    'Gender': '#FF6B6B',
    'Attributes': '#4ECDC4',
    'Demographics': '#45B7D1',
    'Preferences': '#FFA07A',
    'Match': '#98D8C8'
}


class AprioriAnalyzer:
    """
    A comprehensive class for association rule mining on speed dating data.
    
    This class implements the complete pipeline from data preprocessing to
    rule generation, evaluation, visualization, and reporting.
    """
    
    def __init__(self, data_path: str, output_dir: str = OUTPUT_DIR):
        """
        Initialize the Apriori analyzer.
        
        Args:
            data_path: Path to the cleaned CSV file
            output_dir: Directory to save all outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.data: Optional[pd.DataFrame] = None
        self.preprocessed_data: Optional[pd.DataFrame] = None
        self.transactions: Optional[pd.DataFrame] = None
        self.frequent_itemsets: Dict[float, pd.DataFrame] = {}
        self.rules: Optional[pd.DataFrame] = None
        
        self.setup_output_directories()
        self.setup_plotting_style()
    
    def setup_output_directories(self) -> None:
        """Create output directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / 'data',
            self.output_dir / 'visualizations',
            self.output_dir / 'reports'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directories created in: {self.output_dir}")
    
    def setup_plotting_style(self) -> None:
        """Set up consistent plotting style."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': DPI,
            'savefig.dpi': DPI,
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.autolayout': True
        })
    
    def load_data(self) -> None:
        """Load and perform initial validation of the speed dating dataset."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully: {self.data.shape[0]:,} rows, {self.data.shape[1]} columns")
            
            # Display basic info
            print(f"  - Unique participants: {self.data['iid'].nunique():,}")
            print(f"  - Total interactions: {len(self.data):,}")
            print(f"  - Match rate: {self.data['match'].mean():.1%}")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def preprocess_data(self) -> None:
        """
        Preprocess data for Apriori analysis.
        
        This includes:
        - Discretizing continuous variables
        - Creating categorical features
        - Generating derived features
        """
        print("\n🔧 Preprocessing data for association rule mining...")
        
        df = self.data.copy()
        
        # Create race labels
        race_mapping = {
            0: 'Black', 1: 'White', 2: 'Latino', 
            3: 'Asian', 4: 'Native', 6: 'Other'
        }
        df['race_label'] = df['race'].map(race_mapping).fillna('Unknown')
        
        # Discretize ratings given (self-assessment and what they give to partner)
        rating_cols = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
        for col in rating_cols:
            if col in df.columns:
                df[f'{col}_cat'] = pd.cut(
                    df[col], 
                    bins=3, 
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                ).astype(str).replace('nan', 'Unknown')
        
        # Discretize ratings received
        rating_received_cols = ['attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o']
        for col in rating_received_cols:
            if col in df.columns:
                df[f'{col}_cat'] = pd.cut(
                    df[col], 
                    bins=3, 
                    labels=['Low_Rcvd', 'Medium_Rcvd', 'High_Rcvd'],
                    duplicates='drop'
                ).astype(str).replace('nan', 'Unknown')
        
        # Discretize preferences (what they look for)
        pref_cols = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
        for col in pref_cols:
            if col in df.columns:
                df[f'{col}_cat'] = pd.cut(
                    df[col], 
                    bins=3, 
                    labels=['Low_Pref', 'Medium_Pref', 'High_Pref'],
                    duplicates='drop'
                ).astype(str).replace('nan', 'Unknown')
        
        # Discretize age
        if 'age' in df.columns:
            df['age_cat'] = pd.cut(
                df['age'], 
                bins=3, 
                labels=['Young', 'Middle', 'Mature'],
                duplicates='drop'
            ).astype(str).replace('nan', 'Unknown')
        
        # Discretize income
        if 'income' in df.columns:
            # Filter out extreme values
            income_data = df[df['income'] > 0]['income']
            if len(income_data) > 0:
                df['income_cat'] = pd.cut(
                    df['income'], 
                    bins=3, 
                    labels=['Low_Income', 'Medium_Income', 'High_Income'],
                    duplicates='drop'
                ).astype(str).replace('nan', 'Unknown')
        
        # Gender
        df['gender_label'] = df['gender'].map({True: 'Male', False: 'Female'})
        
        # Same race indicator
        if 'samerace' in df.columns:
            df['race_match'] = df['samerace'].map({True: 'Same_Race', False: 'Different_Race'})
        
        # Career - group similar careers
        if 'career' in df.columns:
            # Map to broader categories
            career_mapping = {
                'Lawyer': 'Legal', 'Law': 'Legal',
                'Finance': 'Business', 'Business': 'Business', 'Consulting': 'Business',
                'professor': 'Academic', 'Professor': 'Academic', 'Academic': 'Academic',
                'Scientist': 'STEM', 'Engineer': 'STEM',
                'Social Worker': 'Social_Services',
                'Doctor': 'Healthcare', 'Nurse': 'Healthcare'
            }
            df['career_cat'] = df['career'].map(career_mapping).fillna('Other_Career')
        
        # Create derived features
        
        # Mutual high attraction
        if 'attr' in df.columns and 'attr_o' in df.columns:
            df['mutual_attraction'] = (
                (df['attr'] >= df['attr'].quantile(0.75)) & 
                (df['attr_o'] >= df['attr_o'].quantile(0.75))
            ).map({True: 'Mutual_High_Attr', False: 'Not_Mutual_Attr'})
        
        # Interest alignment (similar ratings for fun and shared interests)
        if 'fun' in df.columns and 'shar' in df.columns:
            df['interest_score'] = (df['fun'] + df['shar']) / 2
            df['interest_alignment'] = pd.cut(
                df['interest_score'],
                bins=3,
                labels=['Low_Interest', 'Medium_Interest', 'High_Interest'],
                duplicates='drop'
            ).astype(str).replace('nan', 'Unknown')
        
        # Expectation met (preference vs received)
        if 'attr1_1' in df.columns and 'attr_o' in df.columns:
            df['attr_expectation_met'] = (
                (df['attr1_1'] <= df['attr_o'])
            ).map({True: 'Attr_Expect_Met', False: 'Attr_Expect_Not_Met'})
        
        # Match outcome
        df['match_outcome'] = df['match'].map({True: 'Match', False: 'No_Match'})
        
        # Decision (did they want to see again)
        if 'dec' in df.columns:
            df['decision'] = df['dec'].map({True: 'Said_Yes', False: 'Said_No'})
        
        self.preprocessed_data = df
        print(f"✓ Preprocessing complete. Created {df.shape[1]} total features.")
    
    def create_transactions(self) -> pd.DataFrame:
        """
        Create transaction format for Apriori algorithm.
        
        Each row becomes a transaction with binary indicators for
        all categorical features.
        
        Returns:
            Binary DataFrame ready for Apriori
        """
        print("\n🛒 Creating transaction format...")
        
        df = self.preprocessed_data.copy()
        
        # Select categorical columns to include
        categorical_features = []
        
        # Gender
        if 'gender_label' in df.columns:
            categorical_features.append('gender_label')
        
        # Age category
        if 'age_cat' in df.columns:
            categorical_features.append('age_cat')
        
        # Race
        if 'race_label' in df.columns:
            categorical_features.append('race_label')
        
        # Career
        if 'career_cat' in df.columns:
            categorical_features.append('career_cat')
        
        # Income
        if 'income_cat' in df.columns:
            categorical_features.append('income_cat')
        
        # Race match
        if 'race_match' in df.columns:
            categorical_features.append('race_match')
        
        # Ratings given (categorized)
        rating_cats = [col for col in df.columns if col.endswith('_cat') and 
                      any(base in col for base in ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar'])]
        categorical_features.extend(rating_cats)
        
        # Derived features
        if 'mutual_attraction' in df.columns:
            categorical_features.append('mutual_attraction')
        if 'interest_alignment' in df.columns:
            categorical_features.append('interest_alignment')
        if 'attr_expectation_met' in df.columns:
            categorical_features.append('attr_expectation_met')
        if 'decision' in df.columns:
            categorical_features.append('decision')
        
        # Match outcome (target)
        if 'match_outcome' in df.columns:
            categorical_features.append('match_outcome')
        
        # Remove duplicates
        categorical_features = list(set(categorical_features))
        
        print(f"  - Selected {len(categorical_features)} categorical features")
        
        # One-hot encode all categorical features
        transactions_list = []
        
        for idx, row in df.iterrows():
            transaction = []
            for feature in categorical_features:
                value = row[feature]
                if pd.notna(value) and value != 'Unknown':
                    # Create item name: Feature_Value
                    item_name = f"{feature}_{value}".replace(' ', '_')
                    transaction.append(item_name)
            transactions_list.append(transaction)
        
        # Convert to binary matrix format
        te = TransactionEncoder()
        te_array = te.fit(transactions_list).transform(transactions_list)
        transactions_df = pd.DataFrame(te_array, columns=te.columns_)
        
        # Remove very rare items (support < 2% to reduce dimensionality)
        item_support = transactions_df.sum() / len(transactions_df)
        frequent_items = item_support[item_support >= 0.02].index
        transactions_df = transactions_df[frequent_items]
        
        self.transactions = transactions_df
        
        print(f"✓ Created transaction matrix: {transactions_df.shape[0]:,} transactions, {transactions_df.shape[1]} items")
        print(f"  - Removed {te_array.shape[1] - transactions_df.shape[1]} rare items (support < 2%)")
        
        return transactions_df
    
    def run_apriori(self, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
        """
        Run Apriori algorithm to find frequent itemsets.
        
        Args:
            min_support: Minimum support threshold
            
        Returns:
            DataFrame of frequent itemsets
        """
        print(f"\n⛏️  Mining frequent itemsets (min_support={min_support})...")
        
        try:
            frequent_itemsets = apriori(
                self.transactions,
                min_support=min_support,
                use_colnames=True,
                max_len=4,  # Limit itemset size for memory efficiency
                low_memory=True,  # Use low memory mode
                verbose=0
            )
            
            print(f"✓ Found {len(frequent_itemsets):,} frequent itemsets")
            
            # Display itemset size distribution
            itemset_sizes = frequent_itemsets['itemsets'].apply(len)
            print(f"  - Itemset size distribution:")
            for size in sorted(itemset_sizes.unique()):
                count = (itemset_sizes == size).sum()
                print(f"    Size {size}: {count:,} itemsets")
            
            return frequent_itemsets
            
        except Exception as e:
            print(f"✗ Error in Apriori: {e}")
            raise
    
    def generate_rules(self, frequent_itemsets: pd.DataFrame, 
                       min_confidence: float = MIN_CONFIDENCE) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.
        
        Args:
            frequent_itemsets: DataFrame of frequent itemsets
            min_confidence: Minimum confidence threshold
            
        Returns:
            DataFrame of association rules with metrics
        """
        print(f"\n📋 Generating association rules (min_confidence={min_confidence})...")
        
        try:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence,
                num_itemsets=len(frequent_itemsets)
            )
            
            if len(rules) == 0:
                print("⚠ No rules found with current thresholds. Try lowering min_confidence.")
                return pd.DataFrame()
            
            # Calculate additional metrics
            
            # Leverage: support(A∪B) - support(A) × support(B)
            rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])
            
            # Zhang's metric (simplified version)
            rules['zhang_metric'] = rules['leverage'] / np.maximum(
                rules['support'] * (1 - rules['antecedent support']),
                rules['antecedent support'] * (1 - rules['support'])
            )
            
            # Format itemsets for readability
            rules['antecedent_str'] = rules['antecedents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules['consequent_str'] = rules['consequents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules['rule'] = rules['antecedent_str'] + ' => ' + rules['consequent_str']
            
            print(f"✓ Generated {len(rules):,} association rules")
            
            # Display metric ranges
            print(f"  - Support range: [{rules['support'].min():.3f}, {rules['support'].max():.3f}]")
            print(f"  - Confidence range: [{rules['confidence'].min():.3f}, {rules['confidence'].max():.3f}]")
            print(f"  - Lift range: [{rules['lift'].min():.3f}, {rules['lift'].max():.3f}]")
            
            return rules
            
        except Exception as e:
            print(f"✗ Error generating rules: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def evaluate_rules(self, rules: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate and filter association rules.
        
        Args:
            rules: DataFrame of association rules
            
        Returns:
            Filtered and ranked DataFrame of rules
        """
        print("\n🔍 Evaluating and filtering rules...")
        
        if len(rules) == 0:
            print("⚠ No rules to evaluate")
            return rules
        
        # Filter out trivial rules (lift < 1.0)
        filtered_rules = rules[rules['lift'] >= 1.0].copy()
        print(f"  - Removed {len(rules) - len(filtered_rules)} rules with lift < 1.0")
        
        # Filter for meaningful lift
        strong_rules = filtered_rules[filtered_rules['lift'] >= MIN_LIFT].copy()
        print(f"  - {len(strong_rules):,} rules with lift >= {MIN_LIFT}")
        
        # Identify rules predicting matches
        match_rules = filtered_rules[
            filtered_rules['consequent_str'].str.contains('match_outcome_Match', case=False)
        ].copy()
        print(f"  - {len(match_rules):,} rules predicting successful matches")
        
        no_match_rules = filtered_rules[
            filtered_rules['consequent_str'].str.contains('match_outcome_No_Match', case=False)
        ].copy()
        print(f"  - {len(no_match_rules):,} rules predicting unsuccessful matches")
        
        # Sort by lift
        filtered_rules = filtered_rules.sort_values('lift', ascending=False).reset_index(drop=True)
        
        print(f"✓ Evaluation complete. {len(filtered_rules):,} rules retained.")
        
        return filtered_rules
    
    def visualize_rules(self, rules: pd.DataFrame) -> None:
        """
        Create comprehensive visualizations of association rules.
        
        Args:
            rules: DataFrame of association rules
        """
        print("\n📊 Creating visualizations...")
        
        if len(rules) == 0:
            print("⚠ No rules to visualize")
            return
        
        viz_dir = self.output_dir / 'visualizations'
        
        # 1. Scatter Plot: Support vs Confidence vs Lift
        self._create_support_confidence_lift_scatter(rules, viz_dir)
        
        # 2. Heatmap of Top Rules
        self._create_rules_heatmap(rules, viz_dir)
        
        # 3. Metrics Distribution
        self._create_metrics_distribution(rules, viz_dir)
        
        # 4. Top Patterns Bar Charts
        self._create_top_patterns_bars(rules, viz_dir)
        
        # 5. Association Network Graph
        self._create_association_network(rules, viz_dir)
        
        print("✓ All visualizations created successfully")
    
    def _create_support_confidence_lift_scatter(self, rules: pd.DataFrame, output_dir: Path) -> None:
        """Create interactive scatter plot of support vs confidence with lift as color."""
        # Take top rules by lift for clearer visualization
        top_rules = rules.nlargest(min(200, len(rules)), 'lift')
        
        fig = px.scatter(
            top_rules,
            x='support',
            y='confidence',
            size='conviction',
            color='lift',
            hover_data=['rule', 'support', 'confidence', 'lift', 'conviction'],
            title='Association Rules: Support vs Confidence (colored by Lift)',
            labels={
                'support': 'Support',
                'confidence': 'Confidence',
                'lift': 'Lift',
                'conviction': 'Conviction'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            width=1000,
            height=700,
            font=dict(size=11)
        )
        
        output_path = output_dir / 'support_confidence_lift_scatter.html'
        fig.write_html(str(output_path))
        print(f"  ✓ Scatter plot saved: {output_path.name}")
    
    def _create_rules_heatmap(self, rules: pd.DataFrame, output_dir: Path) -> None:
        """Create heatmap of top rules by metrics."""
        # Select top 20 rules by lift
        top_rules = rules.nlargest(20, 'lift')
        
        # Create metrics matrix
        metrics_data = top_rules[['support', 'confidence', 'lift', 'conviction', 'leverage']].values
        rule_labels = [rule[:60] + '...' if len(rule) > 60 else rule 
                      for rule in top_rules['rule'].values]
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Normalize by column for better visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        metrics_normalized = scaler.fit_transform(metrics_data)
        
        sns.heatmap(
            metrics_normalized,
            annot=False,
            cmap='RdYlGn',
            xticklabels=['Support', 'Confidence', 'Lift', 'Conviction', 'Leverage'],
            yticklabels=rule_labels,
            cbar_kws={'label': 'Normalized Score'},
            ax=ax
        )
        
        ax.set_title('Top 20 Association Rules - Metrics Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Rules')
        
        plt.tight_layout()
        
        output_path = output_dir / 'rules_heatmap.png'
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Heatmap saved: {output_path.name}")
    
    def _create_metrics_distribution(self, rules: pd.DataFrame, output_dir: Path) -> None:
        """Create distribution plots for key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['support', 'confidence', 'lift', 'conviction', 'leverage', 'zhang_metric']
        
        for idx, metric in enumerate(metrics):
            if metric in rules.columns:
                ax = axes[idx]
                
                # Filter out infinite values for visualization
                metric_data = rules[metric].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(metric_data) > 0:
                    # Histogram
                    ax.hist(metric_data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                    ax.set_xlabel(metric.capitalize())
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {metric.capitalize()}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_val = metric_data.mean()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {mean_val:.3f}')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'No finite values', ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Association Rule Metrics Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / 'metrics_distribution.png'
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Metrics distribution saved: {output_path.name}")
    
    def _create_top_patterns_bars(self, rules: pd.DataFrame, output_dir: Path) -> None:
        """Create bar charts showing top patterns."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top antecedents (what leads to outcomes)
        antecedent_counts = {}
        for itemset in rules['antecedents']:
            for item in itemset:
                antecedent_counts[item] = antecedent_counts.get(item, 0) + 1
        
        top_antecedents = sorted(antecedent_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_antecedents:
            items, counts = zip(*top_antecedents)
            items = [item[:40] + '...' if len(item) > 40 else item for item in items]
            
            axes[0].barh(range(len(items)), counts, color='steelblue')
            axes[0].set_yticks(range(len(items)))
            axes[0].set_yticklabels(items)
            axes[0].set_xlabel('Frequency in Rules')
            axes[0].set_title('Top 15 Antecedents (Conditions)', fontweight='bold')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # Top consequents (outcomes)
        consequent_counts = {}
        for itemset in rules['consequents']:
            for item in itemset:
                consequent_counts[item] = consequent_counts.get(item, 0) + 1
        
        top_consequents = sorted(consequent_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_consequents:
            items, counts = zip(*top_consequents)
            items = [item[:40] + '...' if len(item) > 40 else item for item in items]
            
            axes[1].barh(range(len(items)), counts, color='coral')
            axes[1].set_yticks(range(len(items)))
            axes[1].set_yticklabels(items)
            axes[1].set_xlabel('Frequency in Rules')
            axes[1].set_title('Top 15 Consequents (Outcomes)', fontweight='bold')
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Most Frequent Patterns in Association Rules', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / 'top_patterns_bar.png'
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Top patterns chart saved: {output_path.name}")
    
    def _create_association_network(self, rules: pd.DataFrame, output_dir: Path) -> None:
        """Create network graph of strong associations."""
        # Filter for strong rules
        strong_rules = rules[rules['lift'] >= 2.0].nlargest(50, 'lift')
        
        if len(strong_rules) == 0:
            print("  ⚠ No strong rules (lift >= 2.0) for network visualization")
            return
        
        # Create graph
        G = nx.DiGraph()
        
        for _, rule in strong_rules.iterrows():
            for antecedent in rule['antecedents']:
                for consequent in rule['consequents']:
                    G.add_edge(
                        antecedent, 
                        consequent,
                        weight=rule['confidence'],
                        lift=rule['lift']
                    )
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=RANDOM_SEED)
        
        # Prepare edge traces
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=edge[2]['weight'] * 2,
                        color='lightgray'
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Color by category
            if 'gender' in node.lower():
                node_color.append(NETWORK_COLORS['Gender'])
            elif any(attr in node.lower() for attr in ['attr', 'sinc', 'intel', 'fun', 'amb']):
                node_color.append(NETWORK_COLORS['Attributes'])
            elif any(dem in node.lower() for dem in ['race', 'age', 'income', 'career']):
                node_color.append(NETWORK_COLORS['Demographics'])
            elif 'match' in node.lower():
                node_color.append(NETWORK_COLORS['Match'])
            else:
                node_color.append(NETWORK_COLORS['Preferences'])
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=15,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title='Association Rule Network (Lift >= 2.0, Top 50 Rules)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=800
        )
        
        output_path = output_dir / 'association_network.html'
        fig.write_html(str(output_path))
        print(f"  ✓ Network graph saved: {output_path.name}")
    
    def generate_report(self, rules: pd.DataFrame) -> None:
        """
        Generate comprehensive analysis report.
        
        Args:
            rules: DataFrame of association rules
        """
        print("\n📝 Generating analysis report...")
        
        report_dir = self.output_dir / 'reports'
        
        # Generate markdown report
        self._generate_markdown_report(rules, report_dir)
        
        # Export data files
        self._export_data_files(rules)
        
        print("✓ Reports and data files generated successfully")
    
    def _generate_markdown_report(self, rules: pd.DataFrame, output_dir: Path) -> None:
        """Generate detailed markdown report."""
        
        # Calculate statistics
        total_transactions = len(self.transactions) if self.transactions is not None else 0
        total_items = self.transactions.shape[1] if self.transactions is not None else 0
        total_itemsets = sum(len(itemsets) for itemsets in self.frequent_itemsets.values())
        total_rules = len(rules)
        
        # Match prediction rules
        match_rules = rules[rules['consequent_str'].str.contains('match_outcome_Match', case=False)]
        no_match_rules = rules[rules['consequent_str'].str.contains('match_outcome_No_Match', case=False)]
        
        # Top rules by different metrics
        top_by_lift = rules.nlargest(10, 'lift') if len(rules) > 0 else pd.DataFrame()
        top_by_confidence = rules.nlargest(10, 'confidence') if len(rules) > 0 else pd.DataFrame()
        
        report = f"""# Association Rule Analysis - Speed Dating Dataset

## Executive Summary

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Dataset Overview
- **Total Transactions**: {total_transactions:,}
- **Total Items (Features)**: {total_items:,}
- **Frequent Itemsets Found**: {total_itemsets:,}
- **Association Rules Generated**: {total_rules:,}

### Configuration
- **Minimum Support**: {MIN_SUPPORT}
- **Minimum Confidence**: {MIN_CONFIDENCE}
- **Minimum Lift**: {MIN_LIFT}

---

## Top 10 Association Rules by Lift

Rules with highest lift values indicate the strongest associations between attributes.

| # | Rule | Support | Confidence | Lift | Conviction |
|---|------|---------|------------|------|------------|
"""
        
        for idx, (_, row) in enumerate(top_by_lift.iterrows(), 1):
            report += f"| {idx} | {row['rule'][:80]} | {row['support']:.3f} | {row['confidence']:.3f} | {row['lift']:.3f} | {row['conviction']:.3f} |\n"
        
        report += """
---

## Top 10 Association Rules by Confidence

Rules with highest confidence indicate the most reliable predictions.

| # | Rule | Support | Confidence | Lift | Conviction |
|---|------|---------|------------|------|------------|
"""
        
        for idx, (_, row) in enumerate(top_by_confidence.iterrows(), 1):
            report += f"| {idx} | {row['rule'][:80]} | {row['support']:.3f} | {row['confidence']:.3f} | {row['lift']:.3f} | {row['conviction']:.3f} |\n"
        
        report += f"""
---

## Key Insights

### Patterns for Successful Matches
- **Total rules predicting Match**: {len(match_rules):,}
"""
        
        if len(match_rules) > 0:
            top_match_rules = match_rules.nlargest(5, 'lift')
            report += "\n**Top 5 patterns leading to matches:**\n\n"
            for idx, (_, row) in enumerate(top_match_rules.iterrows(), 1):
                report += f"{idx}. {row['antecedent_str']} => Match (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.1%})\n"
        
        report += f"""

### Patterns for Unsuccessful Matches
- **Total rules predicting No Match**: {len(no_match_rules):,}
"""
        
        if len(no_match_rules) > 0:
            top_no_match_rules = no_match_rules.nlargest(5, 'lift')
            report += "\n**Top 5 patterns leading to no match:**\n\n"
            for idx, (_, row) in enumerate(top_no_match_rules.iterrows(), 1):
                report += f"{idx}. {row['antecedent_str']} => No Match (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.1%})\n"
        
        report += """

### Metrics Summary

**Support Statistics:**
"""
        
        if len(rules) > 0:
            report += f"""
- Minimum: {rules['support'].min():.4f}
- Maximum: {rules['support'].max():.4f}
- Mean: {rules['support'].mean():.4f}
- Median: {rules['support'].median():.4f}

**Confidence Statistics:**
- Minimum: {rules['confidence'].min():.4f}
- Maximum: {rules['confidence'].max():.4f}
- Mean: {rules['confidence'].mean():.4f}
- Median: {rules['confidence'].median():.4f}

**Lift Statistics:**
- Minimum: {rules['lift'].min():.4f}
- Maximum: {rules['lift'].max():.4f}
- Mean: {rules['lift'].mean():.4f}
- Median: {rules['lift'].median():.4f}

**Conviction Statistics:**
- Minimum: {rules['conviction'].min():.4f}
- Maximum: {rules['conviction'].max():.4f}
- Mean: {rules['conviction'].mean():.4f}
- Median: {rules['conviction'].median():.4f}
"""
        
        report += """

---

## Files Generated

### Data Files
- `frequent_itemsets.csv` - All frequent itemsets discovered
- `association_rules.csv` - Complete set of association rules
- `top_rules_by_lift.csv` - Top 50 rules ranked by lift
- `match_prediction_rules.csv` - Rules specifically predicting matches

### Visualizations
- `support_confidence_lift_scatter.html` - Interactive scatter plot of rules
- `rules_heatmap.png` - Heatmap of top rules by metrics
- `metrics_distribution.png` - Distribution plots for all metrics
- `top_patterns_bar.png` - Most frequent patterns in rules
- `association_network.html` - Interactive network graph of associations

---

## Interpretation Guide

### Metrics Explained

**Support**: Frequency of the itemset in the dataset
- Higher support = more common pattern
- Range: [0, 1]

**Confidence**: Probability of consequent given antecedent
- How often the rule is correct
- Range: [0, 1]

**Lift**: How much more likely the consequent is when antecedent is present vs random
- Lift > 1: Positive correlation
- Lift = 1: Independence
- Lift < 1: Negative correlation

**Conviction**: Measure of rule dependence
- Higher conviction = stronger dependence
- Range: [0, ∞]

**Leverage**: Difference between observed and expected support
- Positive leverage = positive correlation
- Range: [-1, 1]

---

*Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_path = output_dir / 'apriori_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  ✓ Markdown report saved: {report_path.name}")
    
    def _export_data_files(self, rules: pd.DataFrame) -> None:
        """Export analysis data to CSV files."""
        data_dir = self.output_dir / 'data'
        
        # Export all frequent itemsets
        if self.frequent_itemsets:
            all_itemsets = pd.concat(self.frequent_itemsets.values(), ignore_index=True)
            all_itemsets.to_csv(data_dir / 'frequent_itemsets.csv', index=False)
            print(f"  ✓ Exported: frequent_itemsets.csv")
        
        # Export all rules
        if len(rules) > 0:
            # Select columns for export
            export_cols = ['antecedent_str', 'consequent_str', 'support', 'confidence', 
                          'lift', 'conviction', 'leverage', 'zhang_metric']
            rules[export_cols].to_csv(data_dir / 'association_rules.csv', index=False)
            print(f"  ✓ Exported: association_rules.csv")
            
            # Top rules by lift
            top_rules = rules.nlargest(50, 'lift')
            top_rules[export_cols].to_csv(data_dir / 'top_rules_by_lift.csv', index=False)
            print(f"  ✓ Exported: top_rules_by_lift.csv")
            
            # Match prediction rules
            match_rules = rules[rules['consequent_str'].str.contains('match_outcome', case=False)]
            if len(match_rules) > 0:
                match_rules[export_cols].to_csv(data_dir / 'match_prediction_rules.csv', index=False)
                print(f"  ✓ Exported: match_prediction_rules.csv")
    
    def run_complete_analysis(self) -> None:
        """Execute the complete Apriori analysis pipeline."""
        print("=" * 70)
        print("APRIORI ASSOCIATION RULE MINING - SPEED DATING ANALYSIS")
        print("=" * 70)
        
        try:
            # Step 1: Load data
            print("\n📥 Step 1: Loading data...")
            self.load_data()
            
            # Step 2: Preprocess
            print("\n🔧 Step 2: Preprocessing data...")
            self.preprocess_data()
            
            # Step 3: Create transactions
            print("\n🛒 Step 3: Creating transactions...")
            self.create_transactions()
            
            # Step 4: Run Apriori with multiple thresholds
            print("\n⛏️  Step 4: Mining frequent itemsets...")
            support_thresholds = [0.08, 0.10]  # Reduced thresholds for memory efficiency
            for support in support_thresholds:
                itemsets = self.run_apriori(support)
                self.frequent_itemsets[support] = itemsets
            
            # Step 5: Generate rules (using lower support threshold)
            print("\n📋 Step 5: Generating association rules...")
            if 0.08 in self.frequent_itemsets and len(self.frequent_itemsets[0.08]) > 0:
                self.rules = self.generate_rules(self.frequent_itemsets[0.08])
            else:
                print("⚠ Using alternative support threshold for rule generation")
                available_support = list(self.frequent_itemsets.keys())[0]
                self.rules = self.generate_rules(self.frequent_itemsets[available_support])
            
            # Step 6: Evaluate and filter
            if self.rules is not None and len(self.rules) > 0:
                print("\n🔍 Step 6: Evaluating and filtering rules...")
                self.rules = self.evaluate_rules(self.rules)
                
                # Step 7: Visualize
                print("\n📊 Step 7: Creating visualizations...")
                self.visualize_rules(self.rules)
                
                # Step 8: Generate reports
                print("\n📝 Step 8: Generating reports...")
                self.generate_report(self.rules)
            else:
                print("\n⚠ No rules generated. Analysis completed without visualizations.")
            
            print(f"\n{'='*70}")
            print("✅ ANALYSIS COMPLETE!")
            print(f"{'='*70}")
            print(f"\n📁 All results saved to: {self.output_dir}")
            print(f"\nCheck the following locations:")
            print(f"  • {self.output_dir / 'data'} - Data exports")
            print(f"  • {self.output_dir / 'visualizations'} - Charts and graphs")
            print(f"  • {self.output_dir / 'reports'} - Analysis report")
            
        except Exception as e:
            print(f"\n✗ Error during analysis: {e}")
            traceback.print_exc()
            raise


def main():
    """Main execution function."""
    # Initialize the analyzer
    data_path = "Speed_Dating_Data_Cleaned.csv"
    
    if not Path(data_path).exists():
        print(f"✗ Error: Data file not found: {data_path}")
        print("Please ensure the cleaned dataset is in the current directory.")
        return
    
    # Create analyzer instance
    analyzer = AprioriAnalyzer(data_path)
    
    # Run complete analysis
    try:
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
