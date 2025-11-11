"""
Speed Dating Dataset - Comprehensive Data Cleaning and Preprocessing Script
===========================================================================

This script implements a complete data cleaning pipeline for the Speed Dating dataset
following the comprehensive plan outlined in clean_dataset.prompt.md

Author: Data Cleaning Pipeline
Date: November 10, 2025
Dataset: Speed Dating Data.csv (8,378 observations × 195 variables)
Output: Speed_Dating_Data_Cleaned.csv

Steps Implemented:
1. Environment Setup and Library Imports
2. Data Loading and Initial Exploration
3. Missing Values Analysis and Treatment
4. Duplicate Detection and Removal
5. Outlier Detection and Management
6. Scale Normalization
7. Categorical Variable Encoding
8. Feature Engineering - Derived Attributes
9. Data Type Optimization
10. Final Data Validation
11. Export Cleaned Data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_step(step_number, step_name):
    """Print formatted step header"""
    print("\n" + "="*80)
    print(f"STEP {step_number}: {step_name}")
    print("="*80)

def print_info(message, indent=0):
    """Print formatted info message"""
    prefix = "  " * indent + "➤ "
    print(f"{prefix}{message}")

def calculate_missing_percentage(df):
    """Calculate missing value percentage for each column"""
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_percent.values
    })
    return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[column].iloc[np.where(z_scores > threshold)[0]]
    return outliers

# ============================================================================
# STEP 1: ENVIRONMENT SETUP AND LIBRARY IMPORTS
# ============================================================================

print_step(1, "Environment Setup and Library Imports")
print_info("Libraries imported successfully:")
print_info(f"pandas version: {pd.__version__}", 1)
print_info(f"numpy version: {np.__version__}", 1)
print_info("scikit-learn, scipy imported", 1)

# ============================================================================
# STEP 2: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print_step(2, "Data Loading and Initial Exploration")

# Load the dataset
data_path = "Speed Dating Data.csv"
print_info(f"Loading dataset from: {data_path}")

df = pd.read_csv(data_path, encoding='latin-1', low_memory=False)

print_info(f"Dataset loaded successfully!")
print_info(f"Shape: {df.shape} (rows × columns)", 1)
print_info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", 1)

# Display basic information
print_info("First 3 rows preview:")
print(df.head(3))

print_info("\nColumn data types summary:")
print(df.dtypes.value_counts())

print_info(f"\nNumerical columns: {df.select_dtypes(include=[np.number]).shape[1]}")
print_info(f"Object columns: {df.select_dtypes(include=['object']).shape[1]}")

# ============================================================================
# STEP 3: MISSING VALUES ANALYSIS AND TREATMENT
# ============================================================================

print_step(3, "Missing Values Analysis and Treatment")

# 3.1 Analyze missing patterns
print_info("3.1 Analyzing missing value patterns...")
missing_summary = calculate_missing_percentage(df)
print_info(f"Columns with missing values: {len(missing_summary)}", 1)

# Show top 20 columns with highest missing percentages
print_info("Top 20 columns with highest missing percentages:", 1)
print(missing_summary.head(20))

# Identify columns with >50% missing
high_missing_cols = missing_summary[missing_summary['Missing_Percentage'] > 50]['Column'].tolist()
print_info(f"\nColumns with >50% missing: {len(high_missing_cols)}", 1)
print_info(f"These columns: {high_missing_cols[:10]}...", 2)

# 3.2 Treatment Strategy
print_info("\n3.2 Applying missing value treatment strategies...")

# Create a copy for processing
df_clean = df.copy()

# Convert object columns with numeric values to proper numeric types
print_info("\nConverting string-formatted numeric columns...", 1)
numeric_object_cols = ['income', 'tuition', 'mn_sat', 'zipcode', 'from']
for col in numeric_object_cols:
    if col in df_clean.columns and df_clean[col].dtype == 'object':
        # Remove commas and convert to numeric
        df_clean[col] = pd.to_numeric(df_clean[col].str.replace(',', ''), errors='coerce')
        print_info(f"{col}: Converted from string to numeric", 2)

# Track imputation statistics
imputation_log = []

# A. Demographic Variables
print_info("Treating demographic variables...", 1)

# Age - impute with median by gender
if 'age' in df_clean.columns and df_clean['age'].isnull().sum() > 0:
    age_missing_count = df_clean['age'].isnull().sum()
    df_clean['age_missing'] = df_clean['age'].isnull().astype(int)
    for gender in df_clean['gender'].unique():
        if pd.notna(gender):
            median_age = df_clean[df_clean['gender'] == gender]['age'].median()
            df_clean.loc[(df_clean['gender'] == gender) & (df_clean['age'].isnull()), 'age'] = median_age
    imputation_log.append(f"Age: Imputed {age_missing_count} values with median by gender")
    print_info(f"Age: Imputed {age_missing_count} missing values with gender-specific median", 2)

# Age of partner
if 'age_o' in df_clean.columns and df_clean['age_o'].isnull().sum() > 0:
    age_o_missing = df_clean['age_o'].isnull().sum()
    df_clean['age_o_missing'] = df_clean['age_o'].isnull().astype(int)
    df_clean['age_o'].fillna(df_clean['age_o'].median(), inplace=True)
    imputation_log.append(f"Age_o: Imputed {age_o_missing} values with median")
    print_info(f"Age_o: Imputed {age_o_missing} missing values", 2)

# Race - create Unknown category (will handle in encoding step)
if 'race' in df_clean.columns:
    race_missing = df_clean['race'].isnull().sum()
    if race_missing > 0:
        df_clean['race'].fillna(0, inplace=True)  # 0 = Unknown
        imputation_log.append(f"Race: {race_missing} values marked as Unknown")
        print_info(f"Race: {race_missing} missing values marked as Unknown", 2)

if 'race_o' in df_clean.columns:
    race_o_missing = df_clean['race_o'].isnull().sum()
    if race_o_missing > 0:
        df_clean['race_o'].fillna(0, inplace=True)
        imputation_log.append(f"Race_o: {race_o_missing} values marked as Unknown")
        print_info(f"Race_o: {race_o_missing} missing values marked as Unknown", 2)

# Field - impute with mode
if 'field_cd' in df_clean.columns:
    field_missing = df_clean['field_cd'].isnull().sum()
    if field_missing > 0:
        df_clean['field_cd'].fillna(df_clean['field_cd'].mode()[0], inplace=True)
        imputation_log.append(f"Field_cd: Imputed {field_missing} values with mode")
        print_info(f"Field_cd: Imputed {field_missing} missing values with mode", 2)

# Income - impute with median (already converted to numeric above)
if 'income' in df_clean.columns:
    income_missing = df_clean['income'].isnull().sum()
    if income_missing > 0:
        df_clean['income_missing'] = df_clean['income'].isnull().astype(int)
        df_clean['income'].fillna(df_clean['income'].median(), inplace=True)
        imputation_log.append(f"Income: Imputed {income_missing} values with median")
        print_info(f"Income: Imputed {income_missing} missing values with median", 2)

# B. Rating Variables
print_info("Treating rating variables...", 1)

rating_vars = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar',
               'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o',
               'like', 'prob', 'like_o', 'prob_o']

for var in rating_vars:
    if var in df_clean.columns and df_clean[var].isnull().sum() > 0:
        missing_count = df_clean[var].isnull().sum()
        # Impute with median rating
        median_val = df_clean[var].median()
        df_clean[var].fillna(median_val, inplace=True)
        imputation_log.append(f"{var}: Imputed {missing_count} values with median")
        print_info(f"{var}: Imputed {missing_count} missing values", 2)

# C. Preference allocation variables
print_info("Treating preference allocation variables...", 1)

pref_vars = ['pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha']
importance_vars = [col for col in df_clean.columns if 'attr1_1' in col or 'sinc1_1' in col or 
                   'intel1_1' in col or 'fun1_1' in col or 'amb1_1' in col or 'shar1_1' in col]

for var in pref_vars + importance_vars:
    if var in df_clean.columns and df_clean[var].isnull().sum() > 0:
        missing_count = df_clean[var].isnull().sum()
        # For preferences, missing might mean equal weighting
        if var in pref_vars:
            df_clean[var].fillna(100/6, inplace=True)  # Equal weighting
        else:
            df_clean[var].fillna(df_clean[var].median(), inplace=True)
        imputation_log.append(f"{var}: Imputed {missing_count} values")

# D. Lifestyle/Interest Variables
print_info("Treating lifestyle/interest variables...", 1)

interest_vars = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
                 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
                 'movies', 'concerts', 'music', 'shopping', 'yoga']

for var in interest_vars:
    if var in df_clean.columns and df_clean[var].isnull().sum() > 0:
        missing_count = df_clean[var].isnull().sum()
        df_clean[var].fillna(df_clean[var].median(), inplace=True)
        imputation_log.append(f"{var}: Imputed {missing_count} values with median")

# E. Follow-up variables - create indicators
print_info("Handling follow-up variables...", 1)

followup_vars = ['date_3', 'numdat_3', 'num_in_3']
for var in followup_vars:
    if var in df_clean.columns:
        missing_count = df_clean[var].isnull().sum()
        if missing_count > 0:
            df_clean[f'{var}_responded'] = (~df_clean[var].isnull()).astype(int)
            df_clean[var].fillna(0, inplace=True)
            print_info(f"{var}: Created response indicator, filled {missing_count} with 0", 2)

print_info(f"\nTotal imputation operations: {len(imputation_log)}", 1)

# ============================================================================
# STEP 4: DUPLICATE DETECTION AND REMOVAL
# ============================================================================

print_step(4, "Duplicate Detection and Removal")

print_info("4.1 Checking for exact duplicates...")
exact_duplicates = df_clean.duplicated().sum()
print_info(f"Exact duplicate rows: {exact_duplicates}", 1)

if exact_duplicates > 0:
    df_clean = df_clean.drop_duplicates()
    print_info(f"Removed {exact_duplicates} exact duplicate rows", 1)

print_info("\n4.2 Checking for logical duplicates...")
# Check for duplicate date encounters (same iid + pid + wave)
if all(col in df_clean.columns for col in ['iid', 'pid', 'wave']):
    logical_duplicates = df_clean.duplicated(subset=['iid', 'pid', 'wave']).sum()
    print_info(f"Logical duplicates (same iid+pid+wave): {logical_duplicates}", 1)
    
    if logical_duplicates > 0:
        df_clean = df_clean.drop_duplicates(subset=['iid', 'pid', 'wave'], keep='first')
        print_info(f"Removed {logical_duplicates} logical duplicates", 1)

print_info(f"\nFinal shape after duplicate removal: {df_clean.shape}", 1)

# ============================================================================
# STEP 5: OUTLIER DETECTION AND MANAGEMENT
# ============================================================================

print_step(5, "Outlier Detection and Management")

print_info("5.1 Detecting outliers in key numerical variables...")

# Age outliers
if 'age' in df_clean.columns:
    age_outliers = df_clean[(df_clean['age'] < 18) | (df_clean['age'] > 70)]
    print_info(f"Age outliers (<18 or >70): {len(age_outliers)}", 1)
    
    if len(age_outliers) > 0:
        df_clean['age'] = df_clean['age'].clip(18, 70)
        print_info(f"Age values clipped to range [18, 70]", 2)

# Rating scale outliers
rating_columns = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar',
                  'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o']

for col in rating_columns:
    if col in df_clean.columns:
        outliers = df_clean[(df_clean[col] < 0) | (df_clean[col] > 10)]
        if len(outliers) > 0:
            print_info(f"{col}: {len(outliers)} values outside [0, 10]", 2)
            df_clean[col] = df_clean[col].clip(0, 10)

# Income outliers - winsorize at 1st and 99th percentiles
if 'income' in df_clean.columns:
    income_01 = df_clean['income'].quantile(0.01)
    income_99 = df_clean['income'].quantile(0.99)
    income_outliers_count = len(df_clean[(df_clean['income'] < income_01) | (df_clean['income'] > income_99)])
    
    if income_outliers_count > 0:
        print_info(f"Income: Winsorizing {income_outliers_count} extreme values", 1)
        df_clean['income'] = df_clean['income'].clip(income_01, income_99)

# Correlation scores
if 'int_corr' in df_clean.columns:
    corr_outliers = df_clean[(df_clean['int_corr'] < -1) | (df_clean['int_corr'] > 1)]
    if len(corr_outliers) > 0:
        print_info(f"int_corr: {len(corr_outliers)} values outside [-1, 1]", 1)
        df_clean['int_corr'] = df_clean['int_corr'].clip(-1, 1)

print_info("\nOutlier detection and treatment completed", 1)

# ============================================================================
# STEP 6: SCALE NORMALIZATION
# ============================================================================

print_step(6, "Scale Normalization")

print_info("6.1 Normalizing preference allocation variables (100-point to 10-point scale)...")

# Preference allocations (100-point scale)
pref_allocation_vars = ['pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha']

for var in pref_allocation_vars:
    if var in df_clean.columns:
        # Check if values are in 100-point scale
        max_val = df_clean[var].max()
        if max_val > 10:
            print_info(f"{var}: Converting from 100-point to 10-point scale", 1)
            df_clean[var] = df_clean[var] / 10

# Check importance rating variables (may also be on 100-point scale)
importance_patterns = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
                       'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1']

for pattern in importance_patterns:
    if pattern in df_clean.columns:
        max_val = df_clean[pattern].max()
        if max_val > 10:
            print_info(f"{pattern}: Converting from 100-point to 10-point scale", 1)
            df_clean[pattern] = df_clean[pattern] / 10

print_info("\n6.2 Verifying rating variables are on consistent scale...")
rating_check_vars = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
for var in rating_check_vars:
    if var in df_clean.columns:
        min_val, max_val = df_clean[var].min(), df_clean[var].max()
        print_info(f"{var}: range [{min_val:.2f}, {max_val:.2f}]", 1)

print_info("\nScale normalization completed", 1)

# ============================================================================
# STEP 7: CATEGORICAL VARIABLE ENCODING
# ============================================================================

print_step(7, "Categorical Variable Encoding")

print_info("7.1 Verifying binary variables...")

# Gender
if 'gender' in df_clean.columns:
    print_info(f"Gender values: {df_clean['gender'].unique()}", 1)
    print_info(f"Gender distribution: {df_clean['gender'].value_counts().to_dict()}", 1)

# Match
if 'match' in df_clean.columns:
    print_info(f"Match values: {df_clean['match'].unique()}", 1)
    print_info(f"Match distribution: {df_clean['match'].value_counts().to_dict()}", 1)

print_info("\n7.2 One-hot encoding nominal variables...")

# Race encoding
if 'race' in df_clean.columns:
    print_info("Encoding race variable...", 1)
    race_dummies = pd.get_dummies(df_clean['race'], prefix='race', drop_first=False)
    # Rename for clarity
    race_mapping = {
        'race_0': 'race_Unknown',
        'race_1': 'race_Black',
        'race_2': 'race_White', 
        'race_3': 'race_Latino',
        'race_4': 'race_Asian',
        'race_5': 'race_NativeAmerican',
        'race_6': 'race_Other'
    }
    race_dummies = race_dummies.rename(columns={k: v for k, v in race_mapping.items() if k in race_dummies.columns})
    df_clean = pd.concat([df_clean, race_dummies], axis=1)
    print_info(f"Created {len(race_dummies.columns)} race dummy variables", 2)

# Race_o encoding
if 'race_o' in df_clean.columns:
    print_info("Encoding race_o (partner race) variable...", 1)
    race_o_dummies = pd.get_dummies(df_clean['race_o'], prefix='race_o', drop_first=False)
    race_o_mapping = {
        'race_o_0': 'race_o_Unknown',
        'race_o_1': 'race_o_Black',
        'race_o_2': 'race_o_White',
        'race_o_3': 'race_o_Latino',
        'race_o_4': 'race_o_Asian',
        'race_o_5': 'race_o_NativeAmerican',
        'race_o_6': 'race_o_Other'
    }
    race_o_dummies = race_o_dummies.rename(columns={k: v for k, v in race_o_mapping.items() if k in race_o_dummies.columns})
    df_clean = pd.concat([df_clean, race_o_dummies], axis=1)
    print_info(f"Created {len(race_o_dummies.columns)} race_o dummy variables", 2)

# Field encoding
if 'field_cd' in df_clean.columns:
    print_info("Encoding field_cd variable...", 1)
    # Check number of unique fields
    n_fields = df_clean['field_cd'].nunique()
    print_info(f"Number of unique fields: {n_fields}", 2)
    
    if n_fields <= 20:
        field_dummies = pd.get_dummies(df_clean['field_cd'], prefix='field', drop_first=False)
        df_clean = pd.concat([df_clean, field_dummies], axis=1)
        print_info(f"Created {len(field_dummies.columns)} field dummy variables", 2)
    else:
        print_info(f"Too many field categories ({n_fields}), grouping rare categories", 2)
        # Group categories with <1% frequency
        field_counts = df_clean['field_cd'].value_counts(normalize=True)
        rare_fields = field_counts[field_counts < 0.01].index
        df_clean['field_cd_grouped'] = df_clean['field_cd'].apply(
            lambda x: 'Other' if x in rare_fields else x
        )
        field_dummies = pd.get_dummies(df_clean['field_cd_grouped'], prefix='field', drop_first=False)
        df_clean = pd.concat([df_clean, field_dummies], axis=1)
        print_info(f"Created {len(field_dummies.columns)} field dummy variables (with grouping)", 2)

# Goal encoding (ordinal but keep as numeric)
if 'goal' in df_clean.columns:
    print_info("Goal variable (ordinal): keeping as numeric", 1)
    print_info(f"Goal values: {sorted(df_clean['goal'].dropna().unique())}", 2)

print_info("\nCategorical encoding completed", 1)

# ============================================================================
# STEP 8: FEATURE ENGINEERING - DERIVED ATTRIBUTES
# ============================================================================

print_step(8, "Feature Engineering - Derived Attributes")

print_info("8.1 Creating perception gap features...")

# Self-assessment vs partner rating differences
gap_features = []

if all(col in df_clean.columns for col in ['attr_o', 'attr']):
    df_clean['attr_diff'] = df_clean['attr_o'] - df_clean['attr']
    gap_features.append('attr_diff')
    
if all(col in df_clean.columns for col in ['sinc_o', 'sinc']):
    df_clean['sinc_diff'] = df_clean['sinc_o'] - df_clean['sinc']
    gap_features.append('sinc_diff')
    
if all(col in df_clean.columns for col in ['intel_o', 'intel']):
    df_clean['intel_diff'] = df_clean['intel_o'] - df_clean['intel']
    gap_features.append('intel_diff')
    
if all(col in df_clean.columns for col in ['fun_o', 'fun']):
    df_clean['fun_diff'] = df_clean['fun_o'] - df_clean['fun']
    gap_features.append('fun_diff')
    
if all(col in df_clean.columns for col in ['amb_o', 'amb']):
    df_clean['amb_diff'] = df_clean['amb_o'] - df_clean['amb']
    gap_features.append('amb_diff')
    
if all(col in df_clean.columns for col in ['shar_o', 'shar']):
    df_clean['shar_diff'] = df_clean['shar_o'] - df_clean['shar']
    gap_features.append('shar_diff')

print_info(f"Created {len(gap_features)} perception gap features", 1)

print_info("\n8.2 Creating age difference features...")

if all(col in df_clean.columns for col in ['age', 'age_o']):
    df_clean['age_diff'] = abs(df_clean['age'] - df_clean['age_o'])
    print_info("Created age_diff (absolute difference)", 1)
    
    # Categorize age gap
    df_clean['age_gap_category'] = pd.cut(
        df_clean['age_diff'],
        bins=[0, 2, 5, 10, 100],
        labels=['Very_Close', 'Close', 'Moderate', 'Large']
    )
    print_info("Created age_gap_category (binned)", 1)

print_info("\n8.3 Creating preference alignment features...")

# Calculate preference match score
if all(col in df_clean.columns for col in ['pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha',
                                             'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o']):
    df_clean['preference_match_score'] = (
        df_clean['pf_o_att'] * df_clean['attr_o'] +
        df_clean['pf_o_sin'] * df_clean['sinc_o'] +
        df_clean['pf_o_int'] * df_clean['intel_o'] +
        df_clean['pf_o_fun'] * df_clean['fun_o'] +
        df_clean['pf_o_amb'] * df_clean['amb_o'] +
        df_clean['pf_o_sha'] * df_clean['shar_o']
    )
    print_info("Created preference_match_score (weighted alignment)", 1)

print_info("\n8.4 Creating mutual interest indicators...")

if all(col in df_clean.columns for col in ['dec', 'dec_o']):
    df_clean['both_interested'] = ((df_clean['dec'] == 1) & (df_clean['dec_o'] == 1)).astype(int)
    df_clean['one_sided_interest'] = (
        ((df_clean['dec'] == 1) & (df_clean['dec_o'] == 0)) | 
        ((df_clean['dec'] == 0) & (df_clean['dec_o'] == 1))
    ).astype(int)
    print_info("Created both_interested and one_sided_interest indicators", 1)

print_info("\n8.5 Creating aggregate rating features...")

# Average rating given to partner
rating_cols = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
if all(col in df_clean.columns for col in rating_cols):
    df_clean['avg_rating_given'] = df_clean[rating_cols].mean(axis=1)
    df_clean['rating_given_std'] = df_clean[rating_cols].std(axis=1)
    print_info("Created avg_rating_given and rating_given_std", 1)

# Average rating received from partner
rating_o_cols = ['attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o']
if all(col in df_clean.columns for col in rating_o_cols):
    df_clean['avg_rating_received'] = df_clean[rating_o_cols].mean(axis=1)
    df_clean['rating_received_std'] = df_clean[rating_o_cols].std(axis=1)
    print_info("Created avg_rating_received and rating_received_std", 1)

# Rating asymmetry
if all(col in df_clean.columns for col in ['avg_rating_given', 'avg_rating_received']):
    df_clean['rating_asymmetry'] = df_clean['avg_rating_given'] - df_clean['avg_rating_received']
    print_info("Created rating_asymmetry", 1)

print_info("\n8.6 Creating expectation vs reality features...")

if all(col in df_clean.columns for col in ['exphappy', 'satis_2']):
    df_clean['expectation_reality_gap'] = df_clean['exphappy'] - df_clean['satis_2']
    print_info("Created expectation_reality_gap", 1)

if all(col in df_clean.columns for col in ['expnum', 'numdat_2']):
    df_clean['expected_actual_dates_gap'] = df_clean['expnum'] - df_clean['numdat_2']
    print_info("Created expected_actual_dates_gap", 1)

print_info("\n8.7 Creating compatibility features...")

# Same race indicator (already exists as 'samerace', verify)
if 'samerace' in df_clean.columns:
    print_info("samerace indicator already exists", 1)

# Same field indicator
if 'field_cd' in df_clean.columns:
    # Note: We don't have partner's field in standard dataset, skip this
    print_info("Same field indicator: requires partner field data (not available)", 1)

print_info("\nFeature engineering completed", 1)
print_info(f"Total new features created: ~{len(gap_features) + 15}", 1)

# ============================================================================
# STEP 9: DATA TYPE OPTIMIZATION
# ============================================================================

print_step(9, "Data Type Optimization")

print_info("Optimizing data types for memory efficiency...")

original_memory = df_clean.memory_usage(deep=True).sum() / 1024**2

# Integer downcasting
int_cols = df_clean.select_dtypes(include=['int64']).columns
for col in int_cols:
    if col in df_clean.columns:
        col_min, col_max = df_clean[col].min(), df_clean[col].max()
        if col_min >= 0 and col_max <= 255:
            df_clean[col] = df_clean[col].astype('uint8')
        elif col_min >= 0 and col_max <= 65535:
            df_clean[col] = df_clean[col].astype('uint16')
        elif col_min >= -128 and col_max <= 127:
            df_clean[col] = df_clean[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            df_clean[col] = df_clean[col].astype('int16')

print_info(f"Downcast {len(int_cols)} integer columns", 1)

# Float precision reduction
float_cols = df_clean.select_dtypes(include=['float64']).columns
for col in float_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype('float32')

print_info(f"Converted {len(float_cols)} float64 columns to float32", 1)

# Binary to boolean
binary_candidates = ['gender', 'match', 'dec', 'dec_o', 'samerace', 'met', 'met_o']
for col in binary_candidates:
    if col in df_clean.columns:
        unique_vals = df_clean[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            df_clean[col] = df_clean[col].astype('bool')

print_info(f"Converted binary columns to boolean", 1)

# Categorical conversion for low-cardinality object columns
object_cols = df_clean.select_dtypes(include=['object']).columns
for col in object_cols:
    if col in df_clean.columns:
        n_unique = df_clean[col].nunique()
        if n_unique < 50:  # Low cardinality
            df_clean[col] = df_clean[col].astype('category')

print_info(f"Converted low-cardinality object columns to category", 1)

optimized_memory = df_clean.memory_usage(deep=True).sum() / 1024**2
memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100

print_info(f"\nMemory optimization results:", 1)
print_info(f"Original memory: {original_memory:.2f} MB", 2)
print_info(f"Optimized memory: {optimized_memory:.2f} MB", 2)
print_info(f"Reduction: {memory_reduction:.1f}%", 2)

# ============================================================================
# STEP 10: FINAL DATA VALIDATION
# ============================================================================

print_step(10, "Final Data Validation")

print_info("10.1 Quality checks...")

# Check missing values in critical columns
critical_cols = ['iid', 'gender', 'match']
critical_missing = df_clean[critical_cols].isnull().sum()
print_info(f"Missing values in critical columns:", 1)
for col in critical_cols:
    if col in df_clean.columns:
        print_info(f"{col}: {critical_missing[col]}", 2)

# Check rating scales
rating_vars_check = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
print_info("\nRating variable ranges (should be 0-10):", 1)
for var in rating_vars_check:
    if var in df_clean.columns:
        min_val, max_val = df_clean[var].min(), df_clean[var].max()
        in_range = (min_val >= 0) and (max_val <= 10)
        status = "✓" if in_range else "✗"
        print_info(f"{var}: [{min_val:.2f}, {max_val:.2f}] {status}", 2)

# Check gender values
if 'gender' in df_clean.columns:
    gender_vals = df_clean['gender'].unique()
    print_info(f"\nGender unique values: {sorted(gender_vals)}", 1)

# Check match values
if 'match' in df_clean.columns:
    match_vals = df_clean['match'].unique()
    print_info(f"Match unique values: {sorted(match_vals)}", 1)

# Check age range
if 'age' in df_clean.columns:
    age_min, age_max = df_clean['age'].min(), df_clean['age'].max()
    age_valid = (age_min >= 18) and (age_max <= 70)
    status = "✓" if age_valid else "✗"
    print_info(f"\nAge range: [{age_min}, {age_max}] {status}", 1)

# Verify derived features exist
print_info("\n10.2 Verifying derived features...", 1)
derived_features = ['attr_diff', 'age_diff', 'avg_rating_given', 'avg_rating_received', 'rating_asymmetry']
for feat in derived_features:
    exists = feat in df_clean.columns
    status = "✓" if exists else "✗"
    print_info(f"{feat}: {status}", 2)

print_info("\n10.3 Final dataset statistics:", 1)
print_info(f"Final shape: {df_clean.shape}", 2)
print_info(f"Final columns: {len(df_clean.columns)}", 2)
print_info(f"Final rows: {len(df_clean)}", 2)
print_info(f"Total missing values: {df_clean.isnull().sum().sum()}", 2)
print_info(f"Missing value percentage: {(df_clean.isnull().sum().sum() / df_clean.size * 100):.2f}%", 2)

# ============================================================================
# STEP 11: EXPORT CLEANED DATA
# ============================================================================

print_step(11, "Export Cleaned Data")

# Generate timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Export main CSV
output_filename = "Speed_Dating_Data_Cleaned.csv"
print_info(f"Exporting cleaned dataset to: {output_filename}")
df_clean.to_csv(output_filename, index=False, encoding='utf-8')
print_info(f"✓ Exported successfully!", 1)

# Export with timestamp backup
output_filename_timestamp = f"Speed_Dating_Data_Cleaned_{timestamp}.csv"
df_clean.to_csv(output_filename_timestamp, index=False, encoding='utf-8')
print_info(f"✓ Backup with timestamp created: {output_filename_timestamp}", 1)

# Optional: Export to Parquet for efficiency
try:
    output_parquet = "Speed_Dating_Data_Cleaned.parquet"
    df_clean.to_parquet(output_parquet, index=False, engine='pyarrow')
    print_info(f"✓ Parquet format exported: {output_parquet}", 1)
except Exception as e:
    print_info(f"Parquet export skipped (pyarrow not available): {str(e)}", 1)

# Generate cleaning report
print_info("\nGenerating cleaning report...")

report_filename = f"Data_Cleaning_Report_{timestamp}.txt"
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SPEED DATING DATASET - DATA CLEANING REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATASET DIMENSIONS\n")
    f.write("-"*80 + "\n")
    f.write(f"Original shape: {df.shape}\n")
    f.write(f"Cleaned shape: {df_clean.shape}\n")
    f.write(f"Rows removed: {len(df) - len(df_clean)}\n")
    f.write(f"Columns added: {len(df_clean.columns) - len(df.columns)}\n\n")
    
    f.write("MISSING VALUE TREATMENT\n")
    f.write("-"*80 + "\n")
    for log_entry in imputation_log[:20]:  # First 20 entries
        f.write(f"  • {log_entry}\n")
    f.write(f"\n  Total imputation operations: {len(imputation_log)}\n\n")
    
    f.write("MEMORY OPTIMIZATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Original memory usage: {original_memory:.2f} MB\n")
    f.write(f"Optimized memory usage: {optimized_memory:.2f} MB\n")
    f.write(f"Memory reduction: {memory_reduction:.1f}%\n\n")
    
    f.write("DERIVED FEATURES CREATED\n")
    f.write("-"*80 + "\n")
    new_columns = set(df_clean.columns) - set(df.columns)
    for i, col in enumerate(sorted(new_columns)[:30], 1):
        f.write(f"  {i}. {col}\n")
    f.write(f"\n  Total new features: {len(new_columns)}\n\n")
    
    f.write("DATA QUALITY SUMMARY\n")
    f.write("-"*80 + "\n")
    f.write(f"Total missing values: {df_clean.isnull().sum().sum()}\n")
    f.write(f"Missing percentage: {(df_clean.isnull().sum().sum() / df_clean.size * 100):.2f}%\n")
    f.write(f"Duplicate rows removed: {exact_duplicates + logical_duplicates}\n\n")
    
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print_info(f"✓ Cleaning report saved: {report_filename}", 1)

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*80)
print("DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nOutput files created:")
print(f"  1. {output_filename}")
print(f"  2. {output_filename_timestamp}")
print(f"  3. {report_filename}")
print("\nDataset is now ready for analysis and modeling!")
print("="*80 + "\n")
