---
agent: agent
---
# Implementation Plan: Apriori-Decision Tree Integration for Speed Dating Analysis

## Overview

This plan outlines the implementation of a unified analysis system that combines Association Rule Mining (Apriori) with Decision Tree/Random Forest insights. The goal is to validate and interpret tree-based model decisions using association rules, creating a comprehensive understanding of match prediction patterns in the speed dating dataset.

**Key Innovation**: Cross-validate Decision Tree split conditions against Apriori association rules to confirm pattern strength and reliability.

---

## Requirements

### Functional Requirements

1. **FR-1**: Load pre-trained Random Forest and Decision Tree models from `decision_tree_results/models/`
2. **FR-2**: Load Apriori association rules from `apriori_results/data/association_rules.csv`
3. **FR-3**: Extract decision rules from trained Decision Tree model
4. **FR-4**: Map continuous Decision Tree splits to categorical Apriori itemsets
5. **FR-5**: Compare and validate tree rules against association rules
6. **FR-6**: Generate interpretable insights showing rule agreement/disagreement
7. **FR-7**: Export results to CSV and Markdown reports
8. **FR-8**: Create visualizations showing rule correspondence

### Non-Functional Requirements

1. **NFR-1**: Single Python file (`hybrid_analysis.py`) with modular class structure
2. **NFR-2**: Follow PEP 8 style guide and SOLID principles
3. **NFR-3**: Execution time < 5 minutes
4. **NFR-4**: Clear logging and progress indicators
5. **NFR-5**: Comprehensive inline documentation (docstrings)
6. **NFR-6**: Graceful error handling with informative messages

### Data Requirements

1. **DR-1**: Use existing cleaned dataset: `Speed_Dating_Data_Cleaned.csv`
2. **DR-2**: Load serialized models: `decision_tree_model.pkl`, `random_forest_model.pkl`
3. **DR-3**: Load Apriori rules: `association_rules.csv`
4. **DR-4**: Feature list from both analyses (68 numeric features + categorical itemsets)

---

## Implementation Steps

### Step 1: Project Setup and Architecture Design

**File Structure:**
```
hybrid_analysis.py                    # Main implementation file
hybrid_results/
├── README.md                         # Documentation
├── data/
│   ├── decision_tree_rules.csv      # Extracted DT rules
│   ├── rule_mappings.csv            # DT-to-Apriori mappings
│   ├── validated_patterns.csv       # Combined insights
│   └── discrepancies.csv            # Conflicting patterns
├── visualizations/
│   ├── rule_comparison_heatmap.png
│   ├── confidence_comparison.png
│   ├── venn_diagram.html            # Rule overlap
│   └── sankey_diagram.html          # Rule flow visualization
└── reports/
    └── hybrid_analysis_report.md    # Executive summary
```

**Class Architecture:**
```python
class HybridAnalyzer:
    """Main orchestrator for hybrid analysis"""
    
    - ModelLoader
    - RuleExtractor
    - RuleMapper
    - RuleValidator
    - ReportGenerator
    - Visualizer
```

---

### Step 2: Load Pre-trained Models and Data

**Class: `ModelLoader`**

**Responsibilities:**
- Load Decision Tree and Random Forest models from pickle files
- Load Apriori association rules from CSV
- Load and prepare feature data
- Validate data integrity

**Key Methods:**

```python
def load_models(self) -> Tuple[DecisionTreeClassifier, RandomForestClassifier]:
    """
    Load serialized sklearn models from decision_tree_results/models/
    
    Returns:
        Tuple of (decision_tree_model, random_forest_model)
    
    Raises:
        FileNotFoundError: If model files don't exist
        ValueError: If models are invalid
    """
```

```python
def load_apriori_rules(self) -> pd.DataFrame:
    """
    Load association rules from apriori_results/data/
    
    Returns:
        DataFrame with columns: antecedents, consequents, support, 
                                confidence, lift, conviction
    """
```

```python
def load_dataset(self) -> pd.DataFrame:
    """
    Load Speed_Dating_Data_Cleaned.csv
    
    Returns:
        DataFrame with cleaned speed dating data
    """
```

**Implementation Details:**
- Use `joblib.load()` for model deserialization
- Validate model type using `isinstance()`
- Parse frozenset strings in Apriori antecedents/consequents
- Handle missing files with clear error messages

---

### Step 3: Extract Decision Rules from Tree Models

**Class: `RuleExtractor`**

**Responsibilities:**
- Extract interpretable rules from Decision Tree
- Get feature importance from Random Forest
- Convert tree paths to readable conditions
- Identify top predictive splits

**Key Methods:**

```python
def extract_tree_rules(self, tree: DecisionTreeClassifier, 
                       feature_names: List[str], 
                       class_names: List[str]) -> List[Dict]:
    """
    Extract all decision paths from trained tree
    
    Algorithm:
    1. Traverse tree from root to leaves
    2. For each leaf node, capture path conditions
    3. Convert to format: [{feature, threshold, direction, prediction, samples}]
    
    Returns:
        List of rule dictionaries with structure:
        {
            'rule_id': int,
            'conditions': List[str],  # e.g., ['attr > 7.5', 'fun > 6.0']
            'prediction': str,         # 'Match' or 'No Match'
            'confidence': float,       # % samples in leaf with majority class
            'support': float,          # % of total samples reaching this leaf
            'samples': int             # Number of training samples
        }
    """
```

```python
def get_feature_splits(self, tree: DecisionTreeClassifier, 
                       feature_names: List[str]) -> pd.DataFrame:
    """
    Get all feature split points used in tree
    
    Returns:
        DataFrame with columns: [feature, threshold, node_id, samples_left, samples_right]
    """
```

**Implementation Strategy:**
- Use `tree.tree_` attribute to access tree structure
- Recursively traverse using `children_left` and `children_right`
- Track path conditions using stack-based DFS
- Calculate confidence from `tree.value` (sample distribution)
- Filter rules by minimum support threshold (e.g., ≥ 5% of samples)

---

### Step 4: Map Decision Tree Splits to Apriori Categories

**Class: `RuleMapper`**

**Responsibilities:**
- Convert continuous thresholds to categorical itemsets
- Match Decision Tree conditions to Apriori antecedents
- Handle feature name translations
- Create unified rule representation

**Key Methods:**

```python
def categorize_threshold(self, feature: str, threshold: float) -> str:
    """
    Map continuous split to Apriori category
    
    Example:
        Input: feature='attr', threshold=7.5
        Output: 'attr_High'  (if attr > 7.5 represents "High")
        
    Logic:
    - Use domain knowledge thresholds:
      - attr/fun/sinc/intel: High (>7), Medium (4-7), Low (≤4)
      - age: Young (<25), Middle (25-30), Mature (>30)
    - Match to existing Apriori itemset naming convention
    """
```

```python
def map_tree_rule_to_apriori(self, tree_rule: Dict) -> Dict:
    """
    Convert Decision Tree rule to Apriori-compatible format
    
    Example:
        Input: {
            'conditions': ['attr > 7.5', 'fun_o > 6.0', 'shar > 5.0'],
            'prediction': 'Match'
        }
        
        Output: {
            'antecedent_items': {'attr_High', 'fun_o_cat_High_Rcvd', 'shar_High'},
            'consequent_items': {'match_outcome_Match'},
            'original_conditions': [...],
            'mapping_confidence': 0.95  # How well conditions map
        }
    
    Returns:
        Dictionary with mapped itemsets and metadata
    """
```

**Threshold Mapping Strategy:**

| Feature | Low | Medium | High | Apriori Category |
|---------|-----|--------|------|------------------|
| attr, fun, sinc, intel | ≤4 | 4-7 | >7 | `{feature}_Low/Medium/High` |
| attr_o, fun_o, etc. | ≤4 | 4-7 | >7 | `{feature}_cat_Low/Medium/High_Rcvd` |
| age | <25 | 25-30 | >30 | `Age_Young/Middle/Mature` |
| samerace | 0 | - | 1 | `samerace_Different/Same` |

**Edge Cases:**
- Handle features not present in Apriori (skip or flag)
- Deal with multiple thresholds on same feature in one rule
- Map compound conditions (e.g., `5 < attr < 8`)

---

### Step 5: Validate and Compare Rules

**Class: `RuleValidator`**

**Responsibilities:**
- Find matching Apriori rules for each tree rule
- Compare confidence and support metrics
- Identify agreements and contradictions
- Calculate validation scores

**Key Methods:**

```python
def find_matching_apriori_rules(self, mapped_tree_rule: Dict, 
                                 apriori_rules: pd.DataFrame) -> pd.DataFrame:
    """
    Search for Apriori rules with matching antecedent/consequent
    
    Matching Criteria:
    1. Exact match: All tree itemsets appear in Apriori rule
    2. Subset match: Tree itemsets are subset of Apriori antecedent
    3. Overlap match: At least 2 common itemsets
    
    Returns:
        Filtered DataFrame of matching Apriori rules with similarity score
    """
```

```python
def validate_rule(self, tree_rule: Dict, apriori_matches: pd.DataFrame) -> Dict:
    """
    Compare metrics between Decision Tree rule and Apriori rules
    
    Validation Checks:
    1. Confidence Agreement: |DT_conf - Apriori_conf| < 0.15
    2. Direction Agreement: Both predict same outcome
    3. Lift Confirmation: Apriori lift > 1.0 (positive association)
    4. Support Correlation: Both have reasonable support
    
    Returns:
        {
            'validation_status': 'CONFIRMED' | 'PARTIAL' | 'CONFLICTING' | 'NO_MATCH',
            'dt_confidence': float,
            'apriori_confidence': float,
            'confidence_delta': float,
            'apriori_lift': float,
            'agreement_score': float,  # 0-100
            'matching_rules': List[Dict]
        }
    """
```

```python
def calculate_agreement_score(self, tree_rule: Dict, apriori_rule: pd.Series) -> float:
    """
    Compute overall agreement between rules
    
    Factors:
    - Itemset overlap: 40% weight
    - Confidence similarity: 30% weight
    - Support correlation: 15% weight
    - Lift strength: 15% weight
    
    Returns:
        Score from 0-100 (100 = perfect agreement)
    """
```

**Validation Categories:**

1. **CONFIRMED** (Score ≥ 80):
   - Apriori strongly supports tree split
   - High lift, similar confidence
   - Example: Tree says `attr>7.5 → Match (75%)`, Apriori has `{attr_High} → {Match} (conf=0.72, lift=4.3)`

2. **PARTIAL** (Score 50-79):
   - Some support but weaker metrics
   - May have subset of conditions
   - Example: Tree has 3 conditions, Apriori confirms 2 of them

3. **CONFLICTING** (Score 20-49):
   - Contradictory predictions or very different confidence
   - Example: Tree predicts Match, Apriori predicts No Match

4. **NO_MATCH** (Score < 20):
   - No corresponding Apriori rule found
   - Tree uses features not in Apriori analysis

---

### Step 6: Generate Insights and Patterns

**Class: `InsightGenerator`**

**Responsibilities:**
- Identify strongest validated patterns
- Find novel patterns (in tree but not Apriori)
- Highlight contradictions for investigation
- Rank patterns by combined strength

**Key Methods:**

```python
def identify_strongest_patterns(self, validated_rules: List[Dict]) -> pd.DataFrame:
    """
    Rank patterns by combined evidence
    
    Ranking Formula:
    pattern_strength = (
        dt_confidence * 0.3 +
        apriori_confidence * 0.3 +
        apriori_lift * 0.1 +
        agreement_score * 0.3
    )
    
    Returns:
        Top 20 patterns sorted by strength with interpretations
    """
```

```python
def find_novel_tree_patterns(self, tree_rules: List[Dict], 
                             validated_rules: List[Dict]) -> List[Dict]:
    """
    Identify Decision Tree rules with no Apriori support
    
    These patterns may indicate:
    - Nuanced interactions not captured by Apriori min_support
    - Continuous threshold effects
    - Complex feature combinations
    
    Returns:
        List of tree-only patterns worth investigating
    """
```

```python
def analyze_contradictions(self, conflicting_rules: List[Dict]) -> pd.DataFrame:
    """
    Deep dive into rules where tree and Apriori disagree
    
    Analysis:
    - Why might they differ?
    - Check sample sizes
    - Examine feature distributions
    - Identify potential overfitting in tree
    
    Returns:
        DataFrame with contradiction analysis and recommendations
    """
```

---

### Step 7: Create Visualizations

**Class: `Visualizer`**

**Responsibilities:**
- Generate comparative charts
- Show rule overlap and agreement
- Visualize pattern strength
- Create interactive dashboards

**Visualizations to Create:**

1. **Rule Comparison Heatmap**
   ```python
   def plot_rule_comparison_heatmap(self, validated_rules: List[Dict]) -> None:
       """
       2D heatmap showing:
       - X-axis: Decision Tree rules
       - Y-axis: Matching Apriori rules
       - Color: Agreement score
       - Annotations: Confidence values
       """
   ```

2. **Confidence Comparison Scatter**
   ```python
   def plot_confidence_comparison(self, validated_rules: List[Dict]) -> None:
       """
       Scatter plot:
       - X: Decision Tree confidence
       - Y: Apriori confidence
       - Size: Support level
       - Color: Lift value
       - Diagonal line shows perfect agreement
       """
   ```

3. **Venn Diagram (Interactive)**
   ```python
   def create_rule_overlap_venn(self, tree_rules: List, apriori_rules: List) -> None:
       """
       Interactive Plotly Venn:
       - Circle 1: Decision Tree patterns
       - Circle 2: Apriori patterns
       - Overlap: Validated patterns
       - Click to see rule details
       """
   ```

4. **Sankey Flow Diagram**
   ```python
   def create_rule_flow_sankey(self, validated_rules: List[Dict]) -> None:
       """
       Flow visualization:
       - Source: Tree conditions
       - Flow: Agreement strength
       - Target: Apriori confirmation
       - Color: Validation status
       """
   ```

5. **Feature Importance Comparison**
   ```python
   def plot_feature_importance_comparison(self) -> None:
       """
       Side-by-side bar chart:
       - Random Forest feature importance
       - Apriori feature frequency in top rules
       - Highlight agreements/differences
       """
   ```

---

### Step 8: Generate Reports and Export Results

**Class: `ReportGenerator`**

**Responsibilities:**
- Create comprehensive Markdown report
- Export data to CSV files
- Generate executive summary
- Provide actionable recommendations

**Output Files:**

1. **`hybrid_results/data/decision_tree_rules.csv`**
   ```
   Columns: rule_id, conditions, prediction, dt_confidence, dt_support, samples
   ```

2. **`hybrid_results/data/rule_mappings.csv`**
   ```
   Columns: rule_id, tree_condition, apriori_itemset, mapping_confidence
   ```

3. **`hybrid_results/data/validated_patterns.csv`**
   ```
   Columns: pattern_id, tree_rule, apriori_rule, agreement_score, 
            dt_confidence, apriori_confidence, apriori_lift, validation_status
   ```

4. **`hybrid_results/data/discrepancies.csv`**
   ```
   Columns: rule_id, tree_prediction, apriori_prediction, conflict_reason, 
            investigation_notes
   ```

**Report Structure (`hybrid_analysis_report.md`):**

```markdown
# Hybrid Analysis Report: Apriori + Decision Tree Integration

## Executive Summary
- Total tree rules analyzed: X
- Apriori rules loaded: Y
- Validated patterns: Z
- Confirmation rate: W%

## Key Findings

### 1. Strongest Validated Patterns
[Top 10 patterns with full agreement]

### 2. Novel Tree Insights
[Patterns found by tree but not Apriori]

### 3. Contradictions Requiring Investigation
[Rules where methods disagree]

## Detailed Analysis

### Pattern Validation Statistics
[Tables and charts]

### Feature-Level Insights
[Which features are most reliable across methods]

### Recommendations
[Actionable insights for match prediction]
```

---

### Step 9: Main Execution Pipeline

**Class: `HybridAnalyzer` (Main Orchestrator)**

```python
class HybridAnalyzer:
    """
    Main class orchestrating the hybrid analysis
    """
    
    def __init__(self, data_path: str, models_path: str, apriori_path: str):
        """Initialize with paths to data sources"""
        self.logger = self._setup_logging()
        self.loader = ModelLoader(data_path, models_path, apriori_path)
        self.extractor = RuleExtractor()
        self.mapper = RuleMapper()
        self.validator = RuleValidator()
        self.insight_generator = InsightGenerator()
        self.visualizer = Visualizer()
        self.reporter = ReportGenerator()
        
    def run_analysis(self) -> None:
        """
        Execute complete hybrid analysis pipeline
        
        Steps:
        1. Load all data and models
        2. Extract decision tree rules
        3. Map rules to Apriori format
        4. Validate and compare
        5. Generate insights
        6. Create visualizations
        7. Export results and reports
        """
        
        self.logger.info("Starting Hybrid Analysis...")
        
        # Step 1: Load
        dt_model, rf_model, apriori_rules, dataset = self.loader.load_all()
        
        # Step 2: Extract
        tree_rules = self.extractor.extract_tree_rules(dt_model, feature_names)
        feature_splits = self.extractor.get_feature_splits(dt_model, feature_names)
        
        # Step 3: Map
        mapped_rules = [self.mapper.map_tree_rule_to_apriori(rule) 
                       for rule in tree_rules]
        
        # Step 4: Validate
        validated_results = []
        for mapped_rule in mapped_rules:
            matches = self.validator.find_matching_apriori_rules(
                mapped_rule, apriori_rules
            )
            validation = self.validator.validate_rule(mapped_rule, matches)
            validated_results.append(validation)
        
        # Step 5: Insights
        strongest = self.insight_generator.identify_strongest_patterns(validated_results)
        novel = self.insight_generator.find_novel_tree_patterns(tree_rules, validated_results)
        contradictions = self.insight_generator.analyze_contradictions(validated_results)
        
        # Step 6: Visualize
        self.visualizer.plot_rule_comparison_heatmap(validated_results)
        self.visualizer.plot_confidence_comparison(validated_results)
        self.visualizer.create_rule_overlap_venn(tree_rules, apriori_rules)
        self.visualizer.create_rule_flow_sankey(validated_results)
        
        # Step 7: Report
        self.reporter.export_to_csv(validated_results, strongest, novel, contradictions)
        self.reporter.generate_markdown_report(validated_results, strongest, novel)
        
        self.logger.info("Analysis complete! Results saved to hybrid_results/")
```

**Main Entry Point:**

```python
def main():
    """
    Main execution function
    """
    # Configuration
    DATA_PATH = "Speed_Dating_Data_Cleaned.csv"
    MODELS_PATH = "decision_tree_results/models"
    APRIORI_PATH = "apriori_results/data/association_rules.csv"
    OUTPUT_PATH = "hybrid_results"
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/data", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/visualizations", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/reports", exist_ok=True)
    
    # Run analysis
    analyzer = HybridAnalyzer(DATA_PATH, MODELS_PATH, APRIORI_PATH)
    analyzer.run_analysis()
    
    print("\n" + "="*70)
    print("HYBRID ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_PATH}/")
    print(f"- Data exports: {OUTPUT_PATH}/data/")
    print(f"- Visualizations: {OUTPUT_PATH}/visualizations/")
    print(f"- Reports: {OUTPUT_PATH}/reports/")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
```

---

## Testing

### Unit Tests

```python
class TestHybridAnalysis:
    """
    Unit tests for hybrid analysis components
    """
    
    def test_model_loading():
        """Verify models load correctly"""
        
    def test_rule_extraction():
        """Verify tree rules are extracted properly"""
        
    def test_threshold_mapping():
        """Verify continuous thresholds map to categories"""
        
    def test_rule_matching():
        """Verify Apriori rules are matched correctly"""
        
    def test_validation_scoring():
        """Verify agreement scores are calculated correctly"""
```

### Integration Tests

1. **Test with sample rule:**
   - Input: Tree rule `attr > 7.5 AND fun > 6.0 → Match`
   - Expected: Should find Apriori rule `{attr_High, fun_High} → {match_outcome_Match}`
   - Verify: Agreement score > 80

2. **Test edge cases:**
   - Tree rule with no Apriori match
   - Conflicting predictions
   - Multiple matching Apriori rules

3. **Test performance:**
   - Should complete in < 5 minutes
   - Should handle all ~100-200 tree rules
   - Memory usage < 2GB

---

## Dependencies

```python
# requirements.txt additions
scikit-learn>=1.3.0      # Already installed (for model loading)
pandas>=2.0.0            # Already installed
numpy>=1.24.0            # Already installed
matplotlib>=3.7.0        # Already installed
seaborn>=0.12.0          # Already installed
plotly>=5.14.0           # Already installed
joblib>=1.3.0            # Already installed (for model serialization)
networkx>=3.1            # For network-based rule visualization
matplotlib-venn>=0.11    # For Venn diagrams
```

---

## Expected Output Example

**Example Validation Result:**

```
================================================================================
PATTERN VALIDATION: Rule #42
================================================================================

Decision Tree Rule:
  Conditions: attr > 7.5 AND fun_o > 6.0 AND shar > 5.0
  Prediction: Match
  Confidence: 0.73 (73%)
  Support: 0.12 (12% of samples)
  Samples: 1,005

Mapped to Apriori:
  Antecedent: {attr_High, fun_o_cat_High_Rcvd, shar_High}
  Consequent: {match_outcome_Match}

Matching Apriori Rules Found: 2

Best Match:
  Rule: {attr_o_cat_High_Rcvd, fun_o_cat_High_Rcvd} → {match_outcome_Match}
  Confidence: 0.712 (71.2%)
  Lift: 4.33
  Support: 0.102 (10.2%)

Validation Result:
  ✓ STATUS: CONFIRMED
  ✓ Agreement Score: 87.5/100
  ✓ Confidence Delta: 0.018 (within threshold)
  ✓ Lift: 4.33 (strong positive association)
  
Interpretation:
  The Decision Tree's split on attractiveness and fun is STRONGLY SUPPORTED
  by Apriori rules. Both methods independently discovered this pattern with
  similar confidence levels. This is a highly reliable predictor of matches.

================================================================================
```

---

## Performance Optimization

1. **Caching:**
   - Cache model loading (models are large)
   - Cache Apriori rule parsing (frozenset conversion is slow)

2. **Vectorization:**
   - Use pandas operations instead of loops where possible
   - Batch process rule comparisons

3. **Parallel Processing:**
   - Consider `joblib.Parallel` for rule validation (optional)

4. **Memory Management:**
   - Don't load full dataset if only need features
   - Clear intermediate results after use

---

## Success Metrics

1. **Functionality:**
   - ✓ Successfully loads all models and data
   - ✓ Extracts at least 50 decision rules from tree
   - ✓ Maps 90%+ of rules to categorical format
   - ✓ Finds Apriori matches for 60%+ of rules

2. **Quality:**
   - ✓ Agreement scores are meaningful (0-100 scale)
   - ✓ Identifies at least 10 strongly validated patterns
   - ✓ Flags contradictions for investigation

3. **Usability:**
   - ✓ Single command execution
   - ✓ Clear progress logging
   - ✓ Readable output reports
   - ✓ Informative visualizations

4. **Performance:**
   - ✓ Completes in < 5 minutes
   - ✓ Uses < 2GB RAM
   - ✓ No crashes or errors

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Models fail to load | Add model validation checks, clear error messages |
| Feature name mismatch | Create comprehensive feature mapping dictionary |
| No Apriori matches found | Relax matching criteria, use fuzzy matching |
| Memory overflow | Process rules in batches, clear intermediate results |
| Threshold mapping errors | Extensive testing with edge cases, fallback strategies |

---

## Future Enhancements (Out of Scope)

1. Interactive web dashboard (Streamlit/Dash)
2. Real-time rule validation for new data
3. Automated threshold optimization
4. Multi-model ensemble validation
5. Temporal pattern analysis (if dates available)

---

**This plan provides a complete roadmap for implementing a single-file, production-ready hybrid analysis system that combines the interpretability of association rules with the predictive power of decision trees.**
