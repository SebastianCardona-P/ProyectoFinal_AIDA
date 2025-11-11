---
mode: agent
---

# Data Cleaning and Preprocessing Plan - Speed Dating Dataset

## Overview

This plan details a comprehensive approach to clean and preprocess the Speed Dating dataset (2002-2004), which contains 8,378 observations across 195 variables from 21 waves of experimental speed dating events involving 552 participants (277 men, 274 women). The preprocessed data will enable robust analysis of dating preferences and match prediction.

## Requirements

### Functional Requirements

- **Input**: `Speed Dating Data.csv` (8,378 rows × 195 columns)
- **Output**: `Speed_Dating_Data_Cleaned.csv` with preprocessed features
- **Technology Stack**: Python 3.x with pandas, numpy, scikit-learn, scipy
- **Execution**: Single Python script for reproducibility

### Data Quality Requirements

- Handle missing values appropriately based on variable type
- Identify and manage outliers using statistical methods
- Remove duplicate records if any exist
- Ensure data consistency across waves and participants

### Transformation Requirements

- Normalize different rating scales (1-10 vs 100-point scales)
- Encode categorical variables (gender, race, field, goal)
- Create derived features for deeper analysis
- Maintain data integrity throughout transformations

---

## Implementation Steps

### **Step 1: Environment Setup and Library Imports** ⚙️

**Libraries Required:**

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations
- `scikit-learn`: Preprocessing utilities (StandardScaler, LabelEncoder, OneHotEncoder)
- `scipy.stats`: Statistical analysis for outlier detection
- `matplotlib/seaborn`: Optional - for initial data exploration visualization

**Actions:**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from scipy import stats
import warnings
```

---

### **Step 2: Data Loading and Initial Exploration** 📊

**Actions:**

1. Load CSV with appropriate data types
2. Display basic information:
   - Dataset dimensions
   - Column names and types
   - Memory usage
   - First/last rows preview
3. Generate summary statistics for numerical columns
4. Identify columns by type (numerical, categorical, text)

**Key Variables to Understand:**

- **Identifiers**: `iid` (participant ID), `id` (ID within wave), `pid` (partner ID)
- **Outcome Variable**: `match` (binary: mutual interest)
- **Decision Variables**: `dec` (decision to see again), `dec_o` (partner's decision)
- **Rating Variables**: `attr`, `sinc`, `intel`, `fun`, `amb`, `shar` (with variants like `attr_o`, `attr1_1`, etc.)
- **Demographics**: `gender`, `age`, `race`, `race_o`, `field_cd`
- **Preferences**: `pf_o_att`, `pf_o_sin`, etc. (preference allocation)
- **Context**: `wave`, `round`, `order`, `position`

---

### **Step 3: Missing Values Analysis and Treatment** 🔍

**3.1 Identify Missing Patterns**

- Calculate percentage of missing values per column
- Identify columns with >50% missing (consider dropping)
- Analyze if missingness is random or systematic (e.g., missing by wave)

**3.2 Treatment Strategy by Variable Type**

**A. Demographic Variables** (age, race, field_cd, income, etc.):

- **Age**: Impute with median by gender
- **Race/Field**: Mode imputation or create "Unknown" category
- **Income/Zipcode**: Consider median by field/education level
- **Education variables** (undergra, mn_sat, tuition): Group-based imputation

**B. Rating Variables** (attr, sinc, intel, fun, amb, shar and variants):

- Missing ratings likely mean question not answered in that wave
- Options:
  - Forward-fill from previous wave responses if available
  - Impute with participant's mean rating across other dates
  - Use median rating by gender and wave
  - For preference allocations (pf*o*\*): May indicate equal weighting

**C. Lifestyle/Interest Variables** (sports, tvsports, exercise, dining, etc.):

- Missing = not measured in that wave
- Impute with median or mode
- Consider wave-specific imputation

**D. Follow-up Variables** (date_3, numdat_3, satisfaction scores):

- Missing is informative (no follow-up)
- Create binary indicator for participation
- Impute with 0 or separate "No Response" category

**3.3 Implementation**:

- Document imputation strategy for each variable group
- Create indicator columns for significant missingness: `age_missing`, `income_missing`
- Apply imputation methods sequentially

---

### **Step 4: Duplicate Detection and Removal** 🔄

**Actions:**

1. Check for exact duplicates across all columns
2. Check for logical duplicates:
   - Same `iid` + `pid` + `wave` (same date encounter)
   - Verify uniqueness of (iid, wave, round, partner)
3. Investigate suspicious patterns:
   - Identical response patterns from same participant
   - Copy-paste errors in data entry
4. Remove confirmed duplicates, keep first occurrence
5. Log all removals with reasons

---

### **Step 5: Outlier Detection and Management** 📉

**5.1 Identify Outliers**

**Numerical Variables:**

- **Age**: Flag values <18 or >70 (use IQR method)
- **Rating Scales**: Values outside expected ranges (e.g., ratings >10 for 1-10 scales)
- **Preference Allocations**: Should sum to 100
- **Correlation Scores** (int_corr): Should be between -1 and 1
- **Income**: Use log-transformation and IQR

**Methods:**

- **IQR Method**: Q1 - 1.5×IQR to Q3 + 1.5×IQR
- **Z-score**: |z| > 3 for normal distributions
- **Domain Knowledge**: Logical constraints on variables

**5.2 Treatment Strategy**

- **Mild Outliers**: Keep but flag
- **Extreme Outliers**:
  - Invalid ratings: Cap at scale maximum
  - Age anomalies: Replace with median
  - Income: Winsorize at 1st and 99th percentiles
- Create outlier indicator columns if relevant for analysis

---

### **Step 6: Scale Normalization** 📏

**6.1 Identify Scale Inconsistencies**

Different waves used different scales:

- **1-10 Scale**: Most rating variables (attr, sinc, intel, fun, amb, shar)
- **100-Point Scale**: Some preference allocations (pf_o_att, pf_o_sin, etc.)
- **1-3 Scale**: Some binary/ordinal responses
- **Continuous**: Age, income, SAT scores

**6.2 Normalization Strategy**

**A. Standardize Preference Allocations**:

- Convert 100-point scales to 10-point scale: `value / 10`
- Verify: Check that preferences still sum appropriately
- Apply to: `pf_o_att`, `pf_o_sin`, `pf_o_int`, `pf_o_fun`, `pf_o_amb`, `pf_o_sha`
- Apply to corresponding `attr1_1`, `sinc1_1`, etc. variables if on 100-point scale

**B. Standardize Rating Variables**:

- Identify all rating variables and their scales
- Convert any percentage-based ratings to 1-10
- Ensure consistency across: attr, sinc, intel, fun, amb, shar and their variants (\_o, \_1, \_2, \_3, etc.)

**C. Scale Numerical Features for Modeling** (optional, done separately):

- **StandardScaler** (z-score normalization): For features with different units
- **MinMaxScaler** (0-1 range): For features needing bounded scale
- Apply to: age, income, SAT scores, tuition
- **Note**: Keep original values; create new scaled columns

---

### **Step 7: Categorical Variable Encoding** 🏷️

**7.1 Identify Categorical Variables**

- **Binary**: `gender` (0/1), `match`, `dec`, `samerace`
- **Ordinal**: `goal`, `date frequency`, `go_out`
- **Nominal**: `race`, `field`, `field_cd`, `career`, `career_c`
- **Location**: `from`, `zipcode` (high cardinality)

**7.2 Encoding Strategies**

**A. Binary Variables** (Already Encoded):

- `gender`: 0 = Female, 1 = Male (verify and standardize)
- `match`: 0 = No match, 1 = Match
- Keep as-is but ensure consistency

**B. Ordinal Variables** (Label Encoding):

```python
# goal: 1=fun, 2=meet new people, 3=date, 4=serious relationship, 5=say i did it, 6=other
# Already numerically encoded - verify order is meaningful
# date: frequency of dating
# go_out: how often go out
```

- Validate existing encoding
- Create interpretable labels if needed

**C. Nominal Variables** (One-Hot Encoding):

**Race** (race, race_o):

```python
# Typical categories: Black/African American=1, European/Caucasian-American=2,
# Latino/Hispanic=3, Asian/Pacific Islander/Asian-American=4, Native American=5, Other=6
```

- One-hot encode: `race_Black`, `race_White`, `race_Latino`, `race_Asian`, `race_Other`
- Same for `race_o` (partner's race)

**Field** (field_cd):

```python
# Academic field codes (1-18 typical range)
# Major categories like: Law, Economics, Medicine, Engineering, etc.
```

- One-hot encode if ≤20 categories
- Group rare categories (<1%) into "Other"
- Create: `field_Law`, `field_Economics`, `field_Medicine`, etc.

**Career** (career, career_c):

- High cardinality text field
- Group into broader categories (Professional, Business, Academic, Creative, Other)
- Then one-hot encode

**Location** (from, zipcode):

- Too high cardinality for direct encoding
- Options:
  - Extract region/state from text
  - Group zipcodes into regions
  - Create "Same Location" binary feature if both from same place
  - Or drop if not critical

**D. Handling Categorical Missing Values**:

- Create "Unknown" or "Missing" category before encoding
- For one-hot encoding, missing becomes all zeros

---

### **Step 8: Feature Engineering - Derived Attributes** 🔧

**8.1 Self-Assessment vs Partner Rating Differences**

Create perception gap features:

```python
# Attractiveness perception gap
attr_diff = attr_o - attr  # How partner rated me vs how I rated them
attr_self_other_gap = attr3_s - attr_o  # My self-rating vs partner's rating

# Apply for all 6 attributes:
- sinc_diff, sinc_self_other_gap
- intel_diff, intel_self_other_gap
- fun_diff, fun_self_other_gap
- amb_diff, amb_self_other_gap
- shar_diff (shared interests perception)
```

**Interpretation**:

- Positive gap: Partner rated higher than self
- Negative gap: Overestimation
- Useful for understanding compatibility and self-awareness

**8.2 Race Compatibility**

```python
# Already exists as 'samerace' (1=same, 0=different)
# Verify and ensure no missing values

# Additional: Create specific race pairing features
race_pairing = race.astype(str) + '_' + race_o.astype(str)
# Then encode or create binary flags for common pairings
```

**8.3 Age Difference**

```python
age_diff = abs(age - age_o)
age_gap_category = pd.cut(age_diff, bins=[0, 2, 5, 10, 100],
                          labels=['Very Close', 'Close', 'Moderate', 'Large'])
```

**8.4 Preference Alignment**

Compare what participant values vs what partner offers:

```python
# Normalize preferences first, then calculate alignment
# If participant values attractiveness (pf_o_att high) and partner is attractive (attr_o high)
preference_match_attr = pf_o_att * attr_o  # Weight x reality
# Aggregate across all 6 attributes for overall preference match score
total_preference_match = (pf_o_att * attr_o + pf_o_sin * sinc_o +
                          pf_o_int * intel_o + pf_o_fun * fun_o +
                          pf_o_amb * amb_o + pf_o_sha * shar_o)
```

**8.5 Mutual Interest Indicators**

```python
# Decision alignment
both_interested = (dec == 1) & (dec_o == 1)  # Same as match
one_sided_interest = (dec == 1) & (dec_o == 0) | (dec == 0) & (dec_o == 1)

# Rating alignment
rating_correlation = correlation between participant ratings and partner ratings
```

**8.6 Aggregate Rating Features**

```python
# Average rating given to partner
avg_rating_given = (attr + sinc + intel + fun + amb + shar) / 6

# Average rating received from partner
avg_rating_received = (attr_o + sinc_o + intel_o + fun_o + amb_o + shar_o) / 6

# Rating asymmetry
rating_asymmetry = avg_rating_given - avg_rating_received

# Standard deviation of ratings (consistency)
rating_consistency = std([attr, sinc, intel, fun, amb, shar])
```

**8.7 Lifestyle Compatibility**

```python
# Interest overlap score
# Calculate similarity in interests (sports, exercise, dining, museums, etc.)
# Use cosine similarity or simple correlation across interest variables

interest_vars = ['sports', 'tvsports', 'exercise', 'dining', 'museums',
                 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
                 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga']
# Would need both participant's and partner's values
# If available, calculate pairwise similarity
```

**8.8 Expectation vs Reality**

```python
# Expected happiness vs actual satisfaction
expectation_reality_gap = exphappy - satis_2  # (if satis_2 is post-event satisfaction)

# Expected number of dates vs actual
expected_actual_dates_gap = expnum - numdat_2
```

**8.9 Field/Career Compatibility**

```python
same_field = (field_cd == partner_field_cd)  # If partner field available
same_career_category = (career_c == partner_career_c)  # If available
```

**8.10 Importance Ranking**

```python
# Rank attributes by importance for each participant
# Based on attr1_1, sinc1_1, etc. (importance ratings)
most_important_attr = argmax([attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1])
```

---

### **Step 9: Data Type Optimization** 💾

**Actions:**

1. **Integer Downcasting**:
   - Convert int64 → int8/int16 where appropriate (IDs, binary variables)
   - Example: gender, match, wave, round
2. **Float Precision**:
   - Convert float64 → float32 for rating variables (sufficient precision)
3. **Categorical Data Type**:
   - Convert low-cardinality strings to pandas categorical
   - Reduces memory for repeated values
4. **Boolean Conversion**:
   - Binary 0/1 columns → bool type

**Expected Result**: 30-50% memory reduction

---

### **Step 10: Final Data Validation** ✅

**10.1 Quality Checks**

- Verify no remaining missing values in critical columns
- Confirm all scales are consistent (1-10 range)
- Check encoded variables have expected number of categories
- Validate derived features have sensible ranges
- Ensure no data leakage (future information in past records)

**10.2 Statistical Validation**

```python
# Check distributions haven't been severely distorted
# Compare before/after summary statistics
# Verify correlations between key variables make sense
```

**10.3 Documentation**

- Create data dictionary for cleaned dataset
- Document all transformations applied
- Note any assumptions made
- List dropped columns and reasons

---

### **Step 11: Export Cleaned Data** 💾

**Actions:**

1. Save cleaned dataset: `Speed_Dating_Data_Cleaned.csv`
2. Optional: Save in multiple formats:
   - Parquet (efficient, preserves types): `Speed_Dating_Data_Cleaned.parquet`
   - Excel (first 1M rows): `Speed_Dating_Data_Cleaned.xlsx`
3. Generate cleaning report:
   - Original vs cleaned dimensions
   - Missing value treatment summary
   - Outliers handled
   - New features created
   - Memory usage comparison

---

## Testing

### Unit Tests

1. **Missing Value Test**: Verify no critical columns have >5% missing values post-cleaning
2. **Scale Test**: Confirm all rating variables are on consistent 1-10 scale
3. **Encoding Test**: Verify one-hot encoded columns sum correctly
4. **Duplicate Test**: Ensure no duplicate (iid, pid, wave) combinations
5. **Outlier Test**: Check no values outside valid ranges
6. **Type Test**: Verify data types are optimized

### Integration Tests

1. **Pipeline Test**: Run full cleaning script start-to-finish without errors
2. **Reproducibility Test**: Running script twice produces identical output
3. **Validation Test**: Compare row counts at each step

### Data Quality Tests

```python
def test_data_quality(df):
    # No missing in key columns
    assert df[['iid', 'gender', 'match']].isna().sum().sum() == 0

    # Rating scales are valid
    rating_cols = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
    for col in rating_cols:
        assert df[col].between(0, 10).all()

    # Gender is binary
    assert df['gender'].isin([0, 1]).all()

    # Match is binary
    assert df['match'].isin([0, 1]).all()

    # Age is reasonable
    assert df['age'].between(18, 70).all()

    # Derived features exist
    assert 'attr_diff' in df.columns
    assert 'age_diff' in df.columns
    assert 'avg_rating_given' in df.columns

    return "All tests passed!"
```

---

## Expected Output Structure

**Cleaned Dataset Columns (Estimated ~250-300 columns):**

1. **Original Core Variables** (~50 columns preserved as-is or normalized)
2. **Encoded Categorical Variables** (~30-50 one-hot encoded columns)
3. **Derived Features** (~30-40 new features)
4. **Indicator Columns** (~10-20 missingness/outlier flags)

**Sample Final Schema:**

```
ID & Context: iid, id, wave, round, gender, age, partner, pid
Outcome: match, dec, dec_o
Demographics: age, age_o, race_[categories], field_[categories], income_normalized
Ratings (self→partner): attr, sinc, intel, fun, amb, shar
Ratings (partner→self): attr_o, sinc_o, intel_o, fun_o, amb_o, shar_o
Preferences: pf_o_att, pf_o_sin, pf_o_int, pf_o_fun, pf_o_amb, pf_o_sha (normalized)
Derived: attr_diff, sinc_diff, ..., age_diff, samerace, avg_rating_given,
         avg_rating_received, rating_asymmetry, preference_match_score
Interests: sports, exercise, dining, museums, ... (normalized)
Flags: age_missing, income_missing, outlier_flag
```

---

## Implementation Notes

### Code Organization

```python
# Structure of cleaning script
1. Import libraries
2. Define helper functions (imputation, encoding, feature engineering)
3. Load data
4. Step-by-step cleaning (each step in separate function)
5. Data validation
6. Export results
7. Generate report
```

### Best Practices

- **Modular Functions**: Each preprocessing step as a separate function
- **Logging**: Print progress at each step with counts/percentages
- **Backup**: Save intermediate datasets at key checkpoints
- **Vectorization**: Use pandas vectorized operations (avoid loops)
- **Comments**: Document all assumptions and decisions
- **Version Control**: Include script version and timestamp in output filename

### Performance Considerations

- Use `dtype` parameter when loading CSV for faster reading
- Process in chunks if memory issues arise
- Use `.copy()` to avoid SettingWithCopyWarning
- Consider `dask` or `vaex` for very large datasets (though 8K rows is manageable)

---

## Success Criteria

✅ **Completeness**: <2% missing values in final dataset  
✅ **Consistency**: All scales normalized to same range  
✅ **Validity**: All values within expected domains  
✅ **Enrichment**: At least 30 meaningful derived features created  
✅ **Reproducibility**: Script runs without errors and produces identical output  
✅ **Efficiency**: Completes in <5 minutes on standard hardware  
✅ **Documentation**: Clear data dictionary and transformation log

---

## Future Enhancements

1. **Advanced Imputation**: Use KNN or MICE for missing values
2. **Feature Selection**: Apply PCA or feature importance ranking
3. **Automated Testing**: Set up pytest suite
4. **Interactive Validation**: Create Jupyter notebook for exploratory analysis
5. **Pipeline Automation**: Use sklearn Pipeline for streamlined preprocessing

---

## References & Resources

- **Dataset Source**: Speed Dating Experiment 2002-2004
- **Documentation**: Speed Dating Data Key.pdf
- **Libraries**: pandas, numpy, scikit-learn, scipy
- **Methodology**: Standard data science preprocessing best practices

---

**End of Plan** 📋

This comprehensive plan provides a roadmap for transforming the raw speed dating data into a clean, analysis-ready dataset with enhanced features that capture the complexity of human attraction and compatibility.
