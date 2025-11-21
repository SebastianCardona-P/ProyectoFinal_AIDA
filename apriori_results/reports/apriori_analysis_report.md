# Association Rule Analysis - Speed Dating Dataset

## Executive Summary

**Analysis Date**: 2025-11-11 15:53:05

### Dataset Overview
- **Total Transactions**: 8,378
- **Total Items (Features)**: 77
- **Frequent Itemsets Found**: 175,127
- **Association Rules Generated**: 417,107

### Configuration
- **Minimum Support**: 0.08
- **Minimum Confidence**: 0.4
- **Minimum Lift**: 1.2

---

## Top 10 Association Rules by Lift

Rules with highest lift values indicate the strongest associations between attributes.

| # | Rule | Support | Confidence | Lift | Conviction |
|---|------|---------|------------|------|------------|
| 1 | fun_o_cat_High_Rcvd, match_outcome_Match => attr_o_cat_High_Rcvd, decision_Said_ | 0.102 | 0.782 | 4.340 | 3.765 |
| 2 | attr_o_cat_High_Rcvd, decision_Said_Yes => fun_o_cat_High_Rcvd, match_outcome_Ma | 0.102 | 0.566 | 4.340 | 2.005 |
| 3 | match_outcome_Match => attr_o_cat_High_Rcvd, decision_Said_Yes, fun_o_cat_High_R | 0.102 | 0.620 | 4.326 | 2.252 |
| 4 | attr_o_cat_High_Rcvd, decision_Said_Yes, fun_o_cat_High_Rcvd => match_outcome_Ma | 0.102 | 0.712 | 4.326 | 2.905 |
| 5 | amb_o_cat_High_Rcvd, match_outcome_Match => attr_o_cat_High_Rcvd, decision_Said_ | 0.094 | 0.773 | 4.288 | 3.609 |
| 6 | attr_o_cat_High_Rcvd, decision_Said_Yes => amb_o_cat_High_Rcvd, match_outcome_Ma | 0.094 | 0.521 | 4.288 | 1.832 |
| 7 | attr_o_cat_High_Rcvd, decision_Said_Yes => match_outcome_Match, sinc_o_cat_High_ | 0.106 | 0.589 | 4.231 | 2.096 |
| 8 | match_outcome_Match, sinc_o_cat_High_Rcvd => attr_o_cat_High_Rcvd, decision_Said | 0.106 | 0.763 | 4.231 | 3.454 |
| 9 | match_outcome_Match, sinc_o_cat_High_Rcvd => decision_Said_Yes, shar_o_cat_High_ | 0.080 | 0.575 | 4.189 | 2.030 |
| 10 | decision_Said_Yes, shar_o_cat_High_Rcvd => match_outcome_Match, sinc_o_cat_High_ | 0.080 | 0.583 | 4.189 | 2.066 |

---

## Top 10 Association Rules by Confidence

Rules with highest confidence indicate the most reliable predictions.

| # | Rule | Support | Confidence | Lift | Conviction |
|---|------|---------|------------|------|------------|
| 1 | fun_cat_High, shar_cat_High, shar_o_cat_Medium_Rcvd => interest_alignment_High_I | 0.115 | 1.000 | 3.059 | inf |
| 2 | fun_cat_High, gender_label_Male, shar_cat_High => interest_alignment_High_Intere | 0.120 | 1.000 | 3.059 | inf |
| 3 | fun_cat_High, shar_cat_High, shar_o_cat_High_Rcvd => interest_alignment_High_Int | 0.098 | 1.000 | 3.059 | inf |
| 4 | amb_cat_High, fun_cat_High, shar_cat_High => interest_alignment_High_Interest | 0.211 | 1.000 | 3.059 | inf |
| 5 | fun_cat_High, shar1_1_cat_Medium_Pref, shar_cat_High => interest_alignment_High_ | 0.113 | 1.000 | 3.059 | inf |
| 6 | fun_cat_High, shar1_1_cat_Low_Pref, shar_cat_High => interest_alignment_High_Int | 0.119 | 1.000 | 3.059 | inf |
| 7 | amb1_1_cat_Low_Pref, fun_cat_High, shar_cat_High => interest_alignment_High_Inte | 0.201 | 1.000 | 3.059 | inf |
| 8 | fun_cat_High, intel_cat_High, shar_cat_High => interest_alignment_High_Interest | 0.227 | 1.000 | 3.059 | inf |
| 9 | attr_o_cat_Medium_Rcvd, fun_cat_High, shar_cat_High => interest_alignment_High_I | 0.099 | 1.000 | 3.059 | inf |
| 10 | fun_cat_High, gender_label_Female, shar_cat_High => interest_alignment_High_Inte | 0.118 | 1.000 | 3.059 | inf |

---

## Key Insights

### Patterns for Successful Matches
- **Total rules predicting Match**: 306

**Top 5 patterns leading to matches:**

1. attr_o_cat_High_Rcvd, decision_Said_Yes => Match (Lift: 4.34, Confidence: 56.6%)
2. attr_o_cat_High_Rcvd, decision_Said_Yes, fun_o_cat_High_Rcvd => Match (Lift: 4.33, Confidence: 71.2%)
3. attr_o_cat_High_Rcvd, decision_Said_Yes => Match (Lift: 4.29, Confidence: 52.1%)
4. attr_o_cat_High_Rcvd, decision_Said_Yes => Match (Lift: 4.23, Confidence: 58.9%)
5. decision_Said_Yes, shar_o_cat_High_Rcvd => Match (Lift: 4.19, Confidence: 58.3%)


### Patterns for Unsuccessful Matches
- **Total rules predicting No Match**: 28,777

**Top 5 patterns leading to no match:**

1. attr_o_cat_Medium_Rcvd, shar_cat_High => No Match (Lift: 3.11, Confidence: 69.9%)
2. attr_o_cat_Medium_Rcvd, interest_alignment_High_Interest => No Match (Lift: 3.07, Confidence: 62.4%)
3. amb_o_cat_Medium_Rcvd, sinc_o_cat_Medium_Rcvd => No Match (Lift: 3.07, Confidence: 65.6%)
4. intel_o_cat_Medium_Rcvd => No Match (Lift: 3.06, Confidence: 42.4%)
5. interest_alignment_High_Interest => No Match (Lift: 3.06, Confidence: 48.7%)


### Metrics Summary

**Support Statistics:**

- Minimum: 0.0801
- Maximum: 0.8478
- Mean: 0.1769
- Median: 0.1517

**Confidence Statistics:**
- Minimum: 0.4000
- Maximum: 1.0000
- Mean: 0.6418
- Median: 0.6083

**Lift Statistics:**
- Minimum: 1.0000
- Maximum: 4.3402
- Mean: 1.1822
- Median: 1.0782

**Conviction Statistics:**
- Minimum: 1.0000
- Maximum: inf
- Mean: inf
- Median: 1.1444


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

output

✓ Output directories created in: apriori_results
======================================================================
APRIORI ASSOCIATION RULE MINING - SPEED DATING ANALYSIS
======================================================================

📥 Step 1: Loading data...
✓ Data loaded successfully: 8,378 rows, 249 columns
  - Unique participants: 551
  - Total interactions: 8,378
  - Match rate: 16.5%

🔧 Step 2: Preprocessing data...

🔧 Preprocessing data for association rule mining...
✓ Preprocessing complete. Created 279 total features.

🛒 Step 3: Creating transactions...

🛒 Creating transaction format...
  - Selected 29 categorical features
✓ Created transaction matrix: 8,378 transactions, 77 items
  - Removed 11 rare items (support < 2%)

⛏️  Step 4: Mining frequent itemsets...

⛏️  Mining frequent itemsets (min_support=0.08)...
✓ Found 98,832 frequent itemsets
  - Itemset size distribution:
    Size 1: 59 itemsets
    Size 2: 1,282 itemsets
    Size 3: 13,839 itemsets
    Size 4: 83,652 itemsets

⛏️  Mining frequent itemsets (min_support=0.1)...
✓ Found 76,295 frequent itemsets
  - Itemset size distribution:
    Size 1: 56 itemsets
    Size 2: 1,168 itemsets
    Size 3: 11,683 itemsets
    Size 4: 63,388 itemsets

📋 Step 5: Generating association rules...

📋 Generating association rules (min_confidence=0.4)...
✓ Generated 643,840 association rules
  - Support range: [0.080, 0.891]
  - Confidence range: [0.400, 1.000]
  - Lift range: [0.499, 4.340]

🔍 Step 6: Evaluating and filtering rules...

🔍 Evaluating and filtering rules...
  - Removed 226733 rules with lift < 1.0
  - 120,728 rules with lift >= 1.2
  - 306 rules predicting successful matches
  - 28,777 rules predicting unsuccessful matches
✓ Evaluation complete. 417,107 rules retained.

📊 Step 7: Creating visualizations...

📊 Creating visualizations...
  ✓ Scatter plot saved: support_confidence_lift_scatter.html
  ✓ Heatmap saved: rules_heatmap.png
  ✓ Metrics distribution saved: metrics_distribution.png
  ✓ Top patterns chart saved: top_patterns_bar.png
  ✓ Network graph saved: association_network.html
✓ All visualizations created successfully

📝 Step 8: Generating reports...

📝 Generating analysis report...
  ✓ Markdown report saved: apriori_analysis_report.md
  ✓ Exported: frequent_itemsets.csv
  ✓ Exported: association_rules.csv
  ✓ Exported: top_rules_by_lift.csv
  ✓ Exported: match_prediction_rules.csv
✓ Reports and data files generated successfully

======================================================================
✅ ANALYSIS COMPLETE!
======================================================================