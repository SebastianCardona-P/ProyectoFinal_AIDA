---
mode: agent
---
# Implementation Plan: Apriori Association Rule Mining for Speed Dating

## Overview
This plan describes the implementation of a complete association rule mining system using the Apriori algorithm to discover patterns in speed dating data. The goal is to identify combinations of attributes and preferences that lead to successful matches.

## Requirements

### Functional Requirements
1. **Data Preprocessing**
   - Load cleaned data from `Speed_Dating_Data_Cleaned.csv`
   - Convert continuous variables to categorical (intelligent binning)
   - Create transactions in market basket format
   - Filter by successful and unsuccessful matches

2. **Association Rule Generation**
   - Implement Apriori algorithm with configurable thresholds
   - Extract meaningful association rules
   - Calculate multiple metrics: support, confidence, lift, conviction, leverage

3. **Analysis and Evaluation**
   - Rank rules by different metrics
   - Filter redundant rules
   - Identify key patterns for successful matches

4. **Visualization and Reports**
   - Scatter plots (support vs confidence vs lift)
   - Rule heatmaps
   - Association network graphs
   - Detailed report in Markdown and HTML format

### Non-Functional Requirements
1. **Code Quality**
   - Follow SOLID, DRY, KISS principles
   - Type hints in all functions
   - Complete documentation with docstrings
   - Robust error handling

2. **Performance**
   - Efficient processing of large rule sets
   - Use of optimized data structures
   - Parallelization when possible

3. **Usability**
   - Simple execution with a single command
   - Configuration via clear constants
   - Informative progress logs
   - Automatically saved results

## Implementation Steps

### Step 1: Project Structure and Configuration (30 min)

**File**: `apriori_analysis.py`

**Tasks**:
1. Import necessary libraries:
   - `mlxtend.frequent_patterns` (apriori, association_rules)
   - `pandas`, `numpy` for data manipulation
   - `matplotlib`, `seaborn`, `plotly` for visualizations
   - `networkx` for rule graphs
   - `pathlib` for file handling

2. Define configuration constants:
   ```python
   # Apriori thresholds
   MIN_SUPPORT = 0.05
   MIN_CONFIDENCE = 0.3
   MIN_LIFT = 1.2
   
   # Discretization parameters
   N_BINS = 3  # Low, Medium, High
   
   # Output configuration
   OUTPUT_DIR = 'apriori_results'
   ```

3. Create output directory structure

### Step 2: Main Class `AprioriAnalyzer` (45 min)

**Responsibility**: Encapsulate all Apriori analysis logic

**Main methods**:

```python
class AprioriAnalyzer:
    def __init__(self, data_path: str, output_dir: str)
    def load_data(self) -> pd.DataFrame
    def preprocess_data(self) -> pd.DataFrame
    def create_transactions(self) -> pd.DataFrame
    def run_apriori(self, min_support: float) -> pd.DataFrame
    def generate_rules(self, frequent_itemsets: pd.DataFrame) -> pd.DataFrame
    def evaluate_rules(self, rules: pd.DataFrame) -> pd.DataFrame
    def filter_redundant_rules(self, rules: pd.DataFrame) -> pd.DataFrame
    def visualize_rules(self, rules: pd.DataFrame) -> None
    def generate_report(self, rules: pd.DataFrame) -> None
    def run_complete_analysis(self) -> None
```

### Step 3: Data Preprocessing (60 min)

**Function**: `preprocess_data()`

**Tasks**:
1. **Discretization of continuous variables**:
   - Ratings (attr, sinc, intel, fun, amb, shar): → {Low, Medium, High}
   - Received ratings (attr_o, sinc_o, etc.): → {Low, Medium, High}
   - Preferences (attr1_1, sinc1_1, etc.): → {Low_Pref, Medium_Pref, High_Pref}
   - Age: → {Young, Middle, Mature}
   - Income: → {Low_Income, Medium_Income, High_Income}

2. **Existing categorical variables**:
   - Gender: {Male, Female}
   - Race: use created race_label
   - Career: group similar careers
   - Same_race: {Same_Race, Different_Race}

3. **Derived variables**:
   - Attraction_match: If both gave high attractiveness ratings
   - Interest_alignment: If they share interests
   - Expectation_met: If preferences match what was received

4. **Target variable**:
   - Match: {Match, No_Match}

**Output**: DataFrame with all categorized variables

### Step 4: Transaction Creation (45 min)

**Function**: `create_transactions()`

**Tasks**:
1. **Transaction format**:
   - Each row (interaction) → transaction
   - Each categorical column → possible items
   - True/False values for presence/absence

2. **Encoding strategies**:
   - One-hot encoding for all categories
   - Descriptive prefixes: `Gender_Male`, `Attr_Given_High`, `Career_Law`
   
3. **Dataset segmentation**:
   - Transactions of successful matches (match=1)
   - Transactions of failed matches (match=0)
   - Complete transactions for comparison

4. **Optimization**:
   - Remove items with very low frequency (<1%)
   - Convert to efficient boolean format

**Output**: Binary DataFrame (True/False) ready for Apriori

### Step 5: Apriori Execution (30 min)

**Function**: `run_apriori()`

**Tasks**:
1. Apply Apriori algorithm with `mlxtend`:
   ```python
   from mlxtend.frequent_patterns import apriori
   
   frequent_itemsets = apriori(
       transactions_df,
       min_support=MIN_SUPPORT,
       use_colnames=True,
       max_len=5  # Limit itemset size
   )
   ```

2. Execute for different thresholds:
   - Low support (0.03), medium (0.05), high (0.10)
   - Compare number of itemsets found

3. Frequent itemset analysis:
   - Top 20 itemsets by support
   - Distribution of itemset sizes

### Step 6: Rule Generation (45 min)

**Function**: `generate_rules()`

**Tasks**:
1. Generate rules with `mlxtend`:
   ```python
   from mlxtend.frequent_patterns import association_rules
   
   rules = association_rules(
       frequent_itemsets,
       metric="confidence",
       min_threshold=MIN_CONFIDENCE,
       support_only=False
   )
   ```

2. Calculate additional metrics:
   - **Support**: Already calculated by mlxtend
   - **Confidence**: Already calculated
   - **Lift**: Already calculated
   - **Conviction**: Already calculated
   - **Leverage**: `support(A∪B) - support(A) × support(B)`
   - **Zhang's metric**: Dependency measure

3. Format rules for readability:
   - Convert frozensets to readable strings
   - Add interpretable descriptions

### Step 7: Evaluation and Filtering (45 min)

**Function**: `evaluate_rules()`

**Tasks**:
1. **Rule ranking**:
   - Top 50 by lift
   - Top 50 by confidence
   - Top 50 by conviction
   - Balanced rules (lift > 2, confidence > 0.5)

2. **Rule filtering**:
   - Remove trivial rules (consequent = antecedent)
   - Remove rules with lift < 1.0
   - Remove redundant rules (subsets)

3. **Specific analysis for matches**:
   - Rules that predict Match=True
   - Rules that predict Match=False
   - Most influential factors

4. **Semantic interpretation**:
   - Group rules by themes:
     * Gender preferences
     * Demographics and compatibility
     * Personal attributes
     * Expectations vs reality

### Step 8: Visualizations (60 min)

**Function**: `visualize_rules()`

**Charts to generate**:

1. **Scatter Plot: Support vs Confidence vs Lift**
   - Axes: support (x), confidence (y)
   - Color: lift
   - Size: conviction
   - Interactive with Plotly

2. **Heatmap of Top Rules**
   - Rows: Top 20 rules
   - Columns: metrics (support, confidence, lift, conviction)
   - Normalized by column

3. **Association Network Graph**
   - Nodes: frequent items
   - Edges: strong rules (lift > 2)
   - Thickness: confidence
   - Color: category (gender, attributes, demographics)

4. **Metrics Distribution**
   - Histograms of support, confidence, lift
   - Box plots of metrics by rule type

5. **Parallel Coordinates**
   - Visualize multidimensional rules
   - Axes: antecedent, consequent, metrics

6. **Bar Charts of Top Patterns**
   - Most frequent items
   - Most common item pairs
   - Specific patterns for matches

### Step 9: Report Generation (45 min)

**Function**: `generate_report()`

**Reports to generate**:

1. **Markdown Report** (`apriori_analysis_report.md`):
   ```markdown
   # Association Rule Analysis - Speed Dating
   
   ## Executive Summary
   - Total transactions analyzed
   - Frequent itemsets found
   - Rules generated
   - Threshold configuration
   
   ## Top 10 Rules by Lift
   [Formatted table]
   
   ## Top 10 Rules by Confidence
   [Formatted table]
   
   ## Key Insights
   ### Patterns for Successful Matches
   ### Patterns for Failed Matches
   ### Differences by Gender
   ### Role of Demographics
   
   ## Detailed Metrics
   [Descriptive statistics]
   ```

2. **Interactive HTML Report**:
   - Sortable and filterable tables
   - Embedded interactive charts
   - Section navigation

3. **Data export**:
   - `frequent_itemsets.csv`: Frequent itemsets
   - `association_rules.csv`: All rules
   - `top_rules_by_lift.csv`: Top rules
   - `match_prediction_rules.csv`: Rules for predicting matches

### Step 10: Complete Pipeline (30 min)

**Function**: `run_complete_analysis()`

**Execution flow**:
```python
def run_complete_analysis(self):
    print("=" * 60)
    print("APRIORI ASSOCIATION RULE MINING - SPEED DATING")
    print("=" * 60)
    
    # 1. Load data
    print("\n📥 Step 1: Loading data...")
    self.load_data()
    
    # 2. Preprocess
    print("\n🔧 Step 2: Preprocessing data...")
    self.preprocess_data()
    
    # 3. Create transactions
    print("\n🛒 Step 3: Creating transactions...")
    transactions = self.create_transactions()
    
    # 4. Run Apriori with multiple thresholds
    print("\n⛏️ Step 4: Mining frequent itemsets...")
    results = {}
    for support in [0.03, 0.05, 0.10]:
        itemsets = self.run_apriori(support)
        results[support] = itemsets
    
    # 5. Generate rules
    print("\n📋 Step 5: Generating association rules...")
    rules = self.generate_rules(results[0.05])
    
    # 6. Evaluate and filter
    print("\n🔍 Step 6: Evaluating and filtering rules...")
    filtered_rules = self.evaluate_rules(rules)
    
    # 7. Visualize
    print("\n📊 Step 7: Creating visualizations...")
    self.visualize_rules(filtered_rules)
    
    # 8. Generate reports
    print("\n📝 Step 8: Generating reports...")
    self.generate_report(filtered_rules)
    
    print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
```

## Testing

### Test Cases

1. **Data Loading Test**:
   - Verify CSV loads correctly
   - Validate number of rows and columns

2. **Preprocessing Test**:
   - Verify correct binning
   - Validate no null values
   - Check category distribution

3. **Transaction Test**:
   - Verify binary format
   - Validate sum of True per row is reasonable
   - Check no empty columns

4. **Apriori Test**:
   - Verify itemsets are generated
   - Validate support thresholds are respected
   - Check itemsets are valid

5. **Rules Test**:
   - Verify rules are generated
   - Validate metrics (0 ≤ confidence ≤ 1, lift ≥ 0)
   - Check no duplicates

6. **Visualization Test**:
   - Verify files are created
   - Validate charts have data

## File Structure

```
ProyectoFinal_AIDA/
├── Speed_Dating_Data_Cleaned.csv
├── visualize_speed_dating.py
├── apriori_analysis.py  # ← NEW FILE
└── apriori_results/     # ← NEW DIRECTORY
    ├── data/
    │   ├── frequent_itemsets.csv
    │   ├── association_rules.csv
    │   ├── top_rules_by_lift.csv
    │   └── match_prediction_rules.csv
    ├── visualizations/
    │   ├── support_confidence_lift_scatter.html
    │   ├── rules_heatmap.png
    │   ├── association_network.html
    │   ├── metrics_distribution.png
    │   └── top_patterns_bar.png
    └── reports/
        ├── apriori_analysis_report.md
        └── apriori_analysis_report.html
```

## Best Practices Applied

### SOLID Principles

1. **Single Responsibility**: 
   - `AprioriAnalyzer` class focused only on Apriori analysis
   - Small methods with single responsibility

2. **Open/Closed**: 
   - Easy to extend with new metrics without modifying existing code
   - Configuration via constants

3. **Liskov Substitution**: 
   - Correct use of inheritance if extended

4. **Interface Segregation**: 
   - Well-defined public methods
   - Private methods for internal implementation

5. **Dependency Inversion**: 
   - Injected dependencies (data_path, output_dir)
   - No hardcoded dependencies

### DRY (Don't Repeat Yourself)

- Reusable functions for discretization
- Common method for saving visualizations
- Templates for reports

### KISS (Keep It Simple, Stupid)

- Clear and readable code
- Comments where necessary
- Avoid over-engineering

### Other Best Practices

1. **Type Hints**: All functions typed
2. **Docstrings**: Complete documentation
3. **Error Handling**: Try-except where appropriate
4. **Logging**: Informative progress messages
5. **Configurability**: Adjustable parameters
6. **Reproducibility**: Fixed random seed

## Execution

**Single command**:
```bash
python apriori_analysis.py
```

**Expected output**:
- Progress logs in console
- Files in `apriori_results/`
- Readable markdown report
- Interactive visualizations

## Success Metrics

1. ✅ Generates at least 100 association rules
2. ✅ Identifies at least 10 interpretable patterns for matches
3. ✅ Lift > 2 in at least 20 rules
4. ✅ Clear and professional visualizations
5. ✅ Complete and readable report
6. ✅ Error-free execution in <5 minutes

---

**Total estimated implementation time**: 6-8 hours

**Next step**: Implement the code following this detailed plan.
