# Speed Dating Visualization Index

## Overview
This document provides an index of all visualizations created for the speed dating data analysis project. The analysis addresses 5 research questions and tests 5 hypotheses about attractiveness, intelligence, sincerity, and desirability in speed dating contexts.

## Dataset Information
- **Source**: Speed_Dating_Data_Cleaned.csv
- **Total Participants**: 551 unique individuals
- **Total Interactions**: 8,378 speed dating encounters
- **Overall Match Rate**: 16.5%
- **Gender Distribution**: Approximately 50/50 split between males and females

## Visualization Categories

### Research Questions (11 visualizations)

#### RQ1: Gender Preferences - "¿Qué busca cada sexo en el sexo opuesto?"
1. **rq1_gender_preferences_bar.png** - Stacked bar chart comparing attribute importance by gender
2. **rq1_gender_preferences_heatmap.png** - Gender preference matrix showing mean preference scores
3. **rq1_gender_preferences_violin.png** - Distribution shapes for each attribute by gender

#### RQ2: Attractiveness and Success - "Relación entre éxito y nivel de atractivo que buscan"
4. **rq2_attractiveness_success_scatter.png** - Scatter plot with regression lines showing attractiveness preference vs success rate
5. **rq2_attractiveness_quartiles_violin.png** - Match success distribution by attractiveness preference quartiles
6. **rq2_attractiveness_correlation_heatmap.png** - Correlation matrix of attractiveness variables and success metrics

#### RQ3: Perceived Attractiveness - "Relación entre éxito y atractivo que creen que busca el sexo opuesto"
7. **rq3_perceived_attractiveness_scatter.png** - Faceted scatter plot by gender showing perceived preferences vs success
8. **rq3_reality_gap_bar.png** - Comparison of actual vs perceived attractiveness importance

#### RQ4: Sincerity and Success - "Relación entre éxito y nivel de sinceridad que buscan"
9. **rq4_sincerity_success_scatter.png** - Scatter plot with regression showing sincerity preference vs success
10. **rq4_sincerity_hexbin.png** - 2D hexbin plot showing sincerity given vs received colored by success rate

#### RQ5: Perceived Sincerity - "Relación entre éxito y sinceridad que creen que busca el sexo opuesto"
11. **rq5_sincerity_perception_gap.png** - Bar chart comparing actual vs perceived sincerity importance

### Hypothesis Testing (7 visualizations)

#### H1: High Attractiveness = More Desirable
12. **h1_attractiveness_boxplot.png** - Match rate distribution by attractiveness quintiles
13. **h1_attractiveness_violin.png** - Attractiveness distribution for matched vs unmatched participants
    - **Statistical Result**: t=24.732, p<0.001 (STRONGLY CONFIRMED)

#### H2: High Intelligence = More Desirable  
14. **h2_intelligence_scatter.png** - Intelligence rating vs match success with correlation statistics

#### H3: Intelligence + Sincerity = More Desirable
15. **h3_intel_sinc_heatmap.png** - 2D heatmap showing desirability surface across intelligence-sincerity combinations

#### H4: Intelligence + Sincerity > Intelligence Alone
16. **h4_intel_vs_intel_sinc.png** - Violin plot comparing high intelligence groups (alone vs combined with sincerity)
    - **Statistical Result**: t=1.585, p=0.113 (NOT CONFIRMED)

#### H5: Men Value Attractiveness More
17. **h5_gender_attractiveness_bar.png** - Side-by-side bar chart of attractiveness preferences by gender
18. **h5_gender_attractiveness_violin.png** - Distribution of attractiveness preferences with statistical annotations
    - **Statistical Result**: t=34.351, p<0.001 (STRONGLY CONFIRMED)

### Dashboard (1 comprehensive visualization)

#### Summary Dashboard
19. **comprehensive_dashboard.png** - Multi-panel (3×3 grid) dashboard summarizing:
    - Gender preference comparisons
    - Success by attractiveness quintiles
    - Intelligence × sincerity interaction
    - Age distribution
    - Overall match rates
    - Correlation matrix
    - Success rate distribution
    - Ratings given vs received
    - Dataset summary statistics

## Key Statistical Findings

### Confirmed Hypotheses:
- **H1**: People with high attractiveness ratings are significantly more desirable (p<0.001)
- **H5**: Men value attractiveness significantly more than women (p<0.001)

### Rejected Hypotheses:
- **H4**: Intelligence combined with sincerity is NOT significantly more desirable than intelligence alone (p=0.113)

## Technical Specifications

### File Formats:
- **Static visualizations**: PNG format at 300 DPI
- **Interactive plots**: HTML format (where applicable)
- **Report**: Markdown format with UTF-8 encoding

### Color Schemes:
- **Gender coding**: Blue (#1f77b4) for Male, Orange (#ff7f0e) for Female
- **Correlation matrices**: Coolwarm colormap
- **Sequential data**: YlOrRd colormap
- **Diverging data**: RdBu_r colormap

### Software Used:
- **Python 3.11**
- **Libraries**: pandas, numpy, matplotlib, seaborn, plotly, scipy
- **Statistical tests**: Independent t-tests, Pearson correlations

## Data Quality Notes

### Binning Challenges:
Several variables (intelligence, sincerity, attractiveness) had limited unique values, requiring robust binning strategies with duplicate handling.

### Missing Data:
Key analysis variables had no missing values, ensuring robust statistical analyses.

### Sample Size:
All statistical tests had adequate sample sizes (n>100 per group) ensuring reliable results.

---

*Visualization index generated on November 11, 2024*
*Total analysis runtime: Approximately 2-3 minutes*