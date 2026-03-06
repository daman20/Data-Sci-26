#!/usr/bin/env python3
"""
Analysis: Human Trafficking and International Tourism
Comparing trafficking victim patterns with tourism arrivals data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path("/Users/amanagrawal/TSA SHIT/Real Data Sci 26/Data-Sci-26")
FIGURES_DIR = BASE_DIR / "Figures" / "claude"

# Load data
print("Loading data...")
trafficking_df = pd.read_csv(BASE_DIR / "CTDC_global_synthetic_data_v2025.csv")
tourism_df = pd.read_csv(BASE_DIR / "Tourism.csv")

print(f"Trafficking data shape: {trafficking_df.shape}")
print(f"Tourism data shape: {tourism_df.shape}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Reshape tourism data from wide to long format
year_cols = [str(y) for y in range(1995, 2024)]
year_cols_present = [c for c in year_cols if c in tourism_df.columns]

tourism_long = tourism_df.melt(
    id_vars=['Country Name', 'Country Code'],
    value_vars=year_cols_present,
    var_name='year',
    value_name='tourism_arrivals'
)
tourism_long['year'] = tourism_long['year'].astype(int)
tourism_long = tourism_long.dropna(subset=['tourism_arrivals'])
tourism_long['tourism_arrivals'] = tourism_long['tourism_arrivals'].astype(float)

# Aggregate trafficking data by country of exploitation and year
trafficking_df['yearOfRegistration'] = pd.to_numeric(trafficking_df['yearOfRegistration'], errors='coerce')
trafficking_agg = trafficking_df.groupby(['CountryOfExploitation', 'yearOfRegistration']).size().reset_index(name='trafficking_cases')
trafficking_agg.columns = ['Country Code', 'year', 'trafficking_cases']

# Merge datasets
merged_df = pd.merge(
    tourism_long,
    trafficking_agg,
    on=['Country Code', 'year'],
    how='inner'
)
print(f"\nMerged data shape: {merged_df.shape}")
print(f"Countries with both data: {merged_df['Country Code'].nunique()}")
print(f"Years covered: {merged_df['year'].min()} - {merged_df['year'].max()}")

# ============================================================================
# FIGURE 1: Overall Correlation between Tourism and Trafficking
# ============================================================================
print("\nGenerating Figure 1: Tourism vs Trafficking Correlation...")

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    merged_df['tourism_arrivals'] / 1e6,
    merged_df['trafficking_cases'],
    alpha=0.5,
    c=merged_df['year'],
    cmap='viridis',
    s=50
)
ax.set_xlabel('International Tourism Arrivals (millions)', fontsize=12)
ax.set_ylabel('Reported Trafficking Cases', fontsize=12)
ax.set_title('Human Trafficking Cases vs International Tourism Arrivals', fontsize=14)
cbar = plt.colorbar(scatter)
cbar.set_label('Year', fontsize=11)

# Add correlation coefficient
valid_data = merged_df[['tourism_arrivals', 'trafficking_cases']].dropna()
r, p = stats.pearsonr(valid_data['tourism_arrivals'], valid_data['trafficking_cases'])
ax.annotate(f'Pearson r = {r:.3f}\np-value = {p:.2e}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_tourism_vs_trafficking_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Correlation: r={r:.3f}, p={p:.2e}")

# ============================================================================
# FIGURE 2: Top Countries Comparison
# ============================================================================
print("\nGenerating Figure 2: Top Countries by Trafficking Cases...")

# Get top 15 countries by total trafficking cases
top_countries = trafficking_agg.groupby('Country Code')['trafficking_cases'].sum().nlargest(15).index.tolist()

top_data = merged_df[merged_df['Country Code'].isin(top_countries)]
top_summary = top_data.groupby('Country Name').agg({
    'trafficking_cases': 'sum',
    'tourism_arrivals': 'mean'
}).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart for trafficking cases
ax1 = axes[0]
top_sorted = top_summary.sort_values('trafficking_cases', ascending=True)
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_sorted)))
ax1.barh(top_sorted['Country Name'], top_sorted['trafficking_cases'], color=colors)
ax1.set_xlabel('Total Reported Trafficking Cases', fontsize=11)
ax1.set_title('Top 15 Countries: Trafficking Cases', fontsize=12)

# Bar chart for tourism arrivals
ax2 = axes[1]
top_sorted_tourism = top_summary.sort_values('tourism_arrivals', ascending=True)
colors2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_sorted_tourism)))
ax2.barh(top_sorted_tourism['Country Name'], top_sorted_tourism['tourism_arrivals'] / 1e6, color=colors2)
ax2.set_xlabel('Avg Annual Tourism Arrivals (millions)', fontsize=11)
ax2.set_title('Top 15 (by trafficking): Tourism Arrivals', fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_top_countries_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 3: Time Trends
# ============================================================================
print("\nGenerating Figure 3: Time Trends...")

yearly_totals = merged_df.groupby('year').agg({
    'trafficking_cases': 'sum',
    'tourism_arrivals': 'sum'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'tab:red'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Total Trafficking Cases', color=color1, fontsize=12)
line1 = ax1.plot(yearly_totals['year'], yearly_totals['trafficking_cases'],
                  color=color1, marker='o', linewidth=2, markersize=6, label='Trafficking Cases')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Total Tourism Arrivals (billions)', color=color2, fontsize=12)
line2 = ax2.plot(yearly_totals['year'], yearly_totals['tourism_arrivals'] / 1e9,
                  color=color2, marker='s', linewidth=2, markersize=6, label='Tourism Arrivals')
ax2.tick_params(axis='y', labelcolor=color2)

ax1.set_title('Global Trends: Trafficking Cases and Tourism Arrivals Over Time', fontsize=14)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_time_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 4: Regional Analysis
# ============================================================================
print("\nGenerating Figure 4: Regional Analysis...")

# Map country codes to regions
region_mapping = {
    'EUR': ['RUS', 'UKR', 'DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'POL', 'NLD', 'BEL', 'ROU', 'CZE', 'GRC', 'PRT', 'HUN', 'SWE', 'AUT', 'BGR', 'DNK', 'FIN', 'SVK', 'NOR', 'IRL', 'HRV', 'BIH', 'SRB', 'SVN', 'LTU', 'LVA', 'EST', 'MDA', 'ALB', 'MKD', 'MNE', 'BLR', 'CHE'],
    'ASIA': ['CHN', 'JPN', 'KOR', 'IND', 'THA', 'VNM', 'PHL', 'IDN', 'MYS', 'SGP', 'MMR', 'KHM', 'NPL', 'BGD', 'PAK', 'LKA', 'TWN', 'HKG', 'MAC'],
    'AMERICAS': ['USA', 'CAN', 'MEX', 'BRA', 'ARG', 'COL', 'PER', 'CHL', 'VEN', 'ECU', 'BOL', 'PRY', 'URY', 'CRI', 'PAN', 'DOM', 'GTM', 'HND', 'SLV', 'NIC', 'CUB', 'JAM', 'HTI'],
    'AFRICA': ['ZAF', 'NGA', 'EGY', 'MAR', 'KEN', 'ETH', 'GHA', 'TZA', 'UGA', 'CMR', 'CIV', 'SEN', 'ZWE', 'MOZ', 'AGO', 'TUN', 'DZA', 'LBY', 'SDN'],
    'OCEANIA': ['AUS', 'NZL', 'FJI', 'PNG'],
    'MENA': ['SAU', 'ARE', 'QAT', 'KWT', 'BHR', 'OMN', 'JOR', 'LBN', 'ISR', 'IRQ', 'IRN', 'TUR', 'SYR', 'YEM']
}

def get_region(code):
    for region, codes in region_mapping.items():
        if code in codes:
            return region
    return 'Other'

merged_df['region'] = merged_df['Country Code'].apply(get_region)

regional_data = merged_df.groupby('region').agg({
    'trafficking_cases': 'sum',
    'tourism_arrivals': 'sum'
}).reset_index()
regional_data = regional_data[regional_data['region'] != 'Other']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart for trafficking
ax1 = axes[0]
colors_pie = plt.cm.Set2(np.linspace(0, 1, len(regional_data)))
ax1.pie(regional_data['trafficking_cases'], labels=regional_data['region'],
        autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax1.set_title('Trafficking Cases by Region', fontsize=12)

# Pie chart for tourism
ax2 = axes[1]
ax2.pie(regional_data['tourism_arrivals'], labels=regional_data['region'],
        autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax2.set_title('Tourism Arrivals by Region', fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '04_regional_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 5: Trafficking Rate per Tourist
# ============================================================================
print("\nGenerating Figure 5: Trafficking Rate Analysis...")

# Calculate rate per million tourists
country_totals = merged_df.groupby(['Country Code', 'Country Name']).agg({
    'trafficking_cases': 'sum',
    'tourism_arrivals': 'sum'
}).reset_index()

country_totals['rate_per_million_tourists'] = (
    country_totals['trafficking_cases'] / (country_totals['tourism_arrivals'] / 1e6)
)

# Filter for countries with meaningful data
country_totals_filtered = country_totals[
    (country_totals['tourism_arrivals'] > 1e6) &
    (country_totals['trafficking_cases'] > 10)
].copy()

top_rates = country_totals_filtered.nlargest(20, 'rate_per_million_tourists')

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_rates)))
bars = ax.barh(top_rates['Country Name'], top_rates['rate_per_million_tourists'], color=colors)
ax.set_xlabel('Trafficking Cases per Million Tourists', fontsize=12)
ax.set_title('Countries with Highest Trafficking Rate Relative to Tourism', fontsize=14)
ax.axvline(x=country_totals_filtered['rate_per_million_tourists'].median(),
           color='black', linestyle='--', linewidth=1, label='Median')
ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_trafficking_rate_per_tourist.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 6: Exploitation Type Distribution
# ============================================================================
print("\nGenerating Figure 6: Exploitation Types...")

# Analyze exploitation types
exploitation_cols = ['isForcedLabour', 'isSexualExploit', 'isOtherExploit']
exploitation_data = trafficking_df[exploitation_cols].apply(pd.to_numeric, errors='coerce')

exploitation_counts = {
    'Forced Labour': exploitation_data['isForcedLabour'].sum(),
    'Sexual Exploitation': exploitation_data['isSexualExploit'].sum(),
    'Other Exploitation': exploitation_data['isOtherExploit'].sum()
}

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
bars = ax.bar(exploitation_counts.keys(), exploitation_counts.values(), color=colors, edgecolor='black')
ax.set_ylabel('Number of Cases', fontsize=12)
ax.set_title('Distribution of Trafficking Exploitation Types', fontsize=14)

for bar, count in zip(bars, exploitation_counts.values()):
    ax.annotate(f'{int(count):,}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_exploitation_types.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 7: Gender and Age Distribution
# ============================================================================
print("\nGenerating Figure 7: Victim Demographics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gender distribution
ax1 = axes[0]
gender_counts = trafficking_df['gender'].value_counts()
colors_gender = ['#FF9999', '#66B2FF', '#99FF99'][:len(gender_counts)]
ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        colors=colors_gender, startangle=90)
ax1.set_title('Trafficking Victims by Gender', fontsize=12)

# Age distribution
ax2 = axes[1]
age_counts = trafficking_df['ageBroad'].value_counts()
colors_age = plt.cm.Purples(np.linspace(0.3, 0.8, len(age_counts)))
ax2.bar(age_counts.index, age_counts.values, color=colors_age, edgecolor='black')
ax2.set_xlabel('Age Group', fontsize=11)
ax2.set_ylabel('Number of Cases', fontsize=11)
ax2.set_title('Trafficking Victims by Age Group', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_victim_demographics.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 8: Heatmap - Correlation Matrix
# ============================================================================
print("\nGenerating Figure 8: Correlation Heatmap...")

# Create country-level summary for correlation analysis
country_summary = merged_df.groupby('Country Code').agg({
    'trafficking_cases': ['sum', 'mean', 'std'],
    'tourism_arrivals': ['sum', 'mean', 'std']
}).reset_index()
country_summary.columns = ['_'.join(col).strip('_') for col in country_summary.columns.values]

# Flatten and compute correlations
corr_data = country_summary[['trafficking_cases_sum', 'trafficking_cases_mean',
                              'tourism_arrivals_sum', 'tourism_arrivals_mean']].dropna()
corr_data.columns = ['Total Trafficking', 'Avg Trafficking', 'Total Tourism', 'Avg Tourism']
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix: Trafficking and Tourism Metrics', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("STATISTICAL ANALYSIS RESULTS")
print("="*60)

# Overall correlation
r_pearson, p_pearson = stats.pearsonr(
    merged_df['tourism_arrivals'].dropna(),
    merged_df['trafficking_cases'].dropna()
)
r_spearman, p_spearman = stats.spearmanr(
    merged_df['tourism_arrivals'].dropna(),
    merged_df['trafficking_cases'].dropna()
)

print(f"\n1. CORRELATION ANALYSIS:")
print(f"   Pearson correlation:  r = {r_pearson:.4f}, p = {p_pearson:.2e}")
print(f"   Spearman correlation: ρ = {r_spearman:.4f}, p = {p_spearman:.2e}")

# Summary statistics
print(f"\n2. DATASET SUMMARY:")
print(f"   Countries analyzed: {merged_df['Country Code'].nunique()}")
print(f"   Years covered: {merged_df['year'].min()} - {merged_df['year'].max()}")
print(f"   Total trafficking cases: {merged_df['trafficking_cases'].sum():,}")
print(f"   Total tourism arrivals: {merged_df['tourism_arrivals'].sum()/1e9:.2f} billion")

# Top destinations
print(f"\n3. TOP 10 EXPLOITATION DESTINATIONS:")
top_destinations = trafficking_agg.groupby('Country Code')['trafficking_cases'].sum().nlargest(10)
for i, (country, cases) in enumerate(top_destinations.items(), 1):
    print(f"   {i:2d}. {country}: {cases:,} cases")

# Regional breakdown
print(f"\n4. REGIONAL DISTRIBUTION:")
regional_summary = merged_df.groupby('region')['trafficking_cases'].sum().sort_values(ascending=False)
for region, cases in regional_summary.items():
    if region != 'Other':
        pct = cases / regional_summary.sum() * 100
        print(f"   {region}: {cases:,} cases ({pct:.1f}%)")

print("\n" + "="*60)
print("Analysis complete! Figures saved to:", FIGURES_DIR)
print("="*60)
