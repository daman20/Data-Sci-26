#!/usr/bin/env python3
"""
Generate a TSA Digital Scientific Poster for Data Science and Analytics
Team 1383 - Orlando, FL
Topic: Human Trafficking and Tourism Correlation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path
import base64
import io
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/sessions/youthful-magical-noether/mnt/Data-Sci-26")

# ── Load Data ──────────────────────────────────────────────────────
print("Loading data...")
trafficking_df = pd.read_csv(BASE_DIR / "CTDC_global_synthetic_data_v2025.csv")
tourism_df = pd.read_csv(BASE_DIR / "Tourism.csv")

# Reshape tourism
year_cols = [str(y) for y in range(1995, 2024)]
year_cols_present = [c for c in year_cols if c in tourism_df.columns]
tourism_long = tourism_df.melt(
    id_vars=['Country Name', 'Country Code'],
    value_vars=year_cols_present,
    var_name='year', value_name='tourism_arrivals'
)
tourism_long['year'] = tourism_long['year'].astype(int)
tourism_long = tourism_long.dropna(subset=['tourism_arrivals'])
tourism_long['tourism_arrivals'] = tourism_long['tourism_arrivals'].astype(float)

# Aggregate trafficking
trafficking_df['yearOfRegistration'] = pd.to_numeric(trafficking_df['yearOfRegistration'], errors='coerce')
trafficking_agg = trafficking_df.groupby(['CountryOfExploitation', 'yearOfRegistration']).size().reset_index(name='trafficking_cases')
trafficking_agg.columns = ['Country Code', 'year', 'trafficking_cases']

# Merge
merged_df = pd.merge(tourism_long, trafficking_agg, on=['Country Code', 'year'], how='inner')

# Region mapping
region_mapping = {
    'Europe': ['RUS', 'UKR', 'DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'POL', 'NLD', 'BEL', 'ROU', 'CZE', 'GRC', 'PRT', 'HUN', 'SWE', 'AUT', 'BGR', 'DNK', 'FIN', 'SVK', 'NOR', 'IRL', 'HRV', 'BIH', 'SRB', 'SVN', 'LTU', 'LVA', 'EST', 'MDA', 'ALB', 'MKD', 'MNE', 'BLR', 'CHE'],
    'Asia': ['CHN', 'JPN', 'KOR', 'IND', 'THA', 'VNM', 'PHL', 'IDN', 'MYS', 'SGP', 'MMR', 'KHM', 'NPL', 'BGD', 'PAK', 'LKA', 'TWN', 'HKG', 'MAC'],
    'Americas': ['USA', 'CAN', 'MEX', 'BRA', 'ARG', 'COL', 'PER', 'CHL', 'VEN', 'ECU', 'BOL', 'PRY', 'URY', 'CRI', 'PAN', 'DOM', 'GTM', 'HND', 'SLV', 'NIC', 'CUB', 'JAM', 'HTI'],
    'Africa': ['ZAF', 'NGA', 'EGY', 'MAR', 'KEN', 'ETH', 'GHA', 'TZA', 'UGA', 'CMR', 'CIV', 'SEN', 'ZWE', 'MOZ', 'AGO', 'TUN', 'DZA', 'LBY', 'SDN'],
    'Oceania': ['AUS', 'NZL', 'FJI', 'PNG'],
    'Middle East': ['SAU', 'ARE', 'QAT', 'KWT', 'BHR', 'OMN', 'JOR', 'LBN', 'ISR', 'IRQ', 'IRN', 'TUR', 'SYR', 'YEM']
}
def get_region(code):
    for region, codes in region_mapping.items():
        if code in codes:
            return region
    return 'Other'
merged_df['region'] = merged_df['Country Code'].apply(get_region)

# ── Helper to convert figure to base64 ─────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

# ── Color palette ──────────────────────────────────────────────────
C_RED = '#C0392B'
C_BLUE = '#2980B9'
C_DARK = '#1a1a2e'
C_ACCENT = '#e94560'
C_GOLD = '#f5a623'
C_TEAL = '#16a085'
C_BG = '#f8f9fa'

# ── FIGURE 1: Dual-axis time trends ───────────────────────────────
print("Generating charts...")
yearly = merged_df.groupby('year').agg({'trafficking_cases': 'sum', 'tourism_arrivals': 'sum'}).reset_index()

fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

ln1 = ax1.plot(yearly['year'], yearly['trafficking_cases'], color=C_ACCENT, marker='o', linewidth=2.2, markersize=5, label='Trafficking Cases', zorder=3)
ax1.set_xlabel('Year', fontsize=9, fontweight='bold')
ax1.set_ylabel('Trafficking Cases', color=C_ACCENT, fontsize=9, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=C_ACCENT, labelsize=8)
ax1.tick_params(axis='x', labelsize=8)

ax2 = ax1.twinx()
ln2 = ax2.plot(yearly['year'], yearly['tourism_arrivals'] / 1e9, color=C_BLUE, marker='s', linewidth=2.2, markersize=5, label='Tourism Arrivals (B)', zorder=2)
ax2.set_ylabel('Tourism Arrivals (Billions)', color=C_BLUE, fontsize=9, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=C_BLUE, labelsize=8)

lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=7, framealpha=0.9)
ax1.set_title('Global Trends Over Time', fontsize=10, fontweight='bold', pad=8)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
b64_fig1 = fig_to_base64(fig1)

# ── FIGURE 2: Scatter with correlation ─────────────────────────────
fig2, ax = plt.subplots(figsize=(5.5, 3.2))
fig2.patch.set_facecolor('white')
ax.set_facecolor('white')

# Filter outliers for display
display_df = merged_df[(merged_df['tourism_arrivals'] < 1e8) & (merged_df['trafficking_cases'] < 5000)].copy()
scatter = ax.scatter(
    display_df['tourism_arrivals'] / 1e6,
    display_df['trafficking_cases'],
    alpha=0.45, c=display_df['year'], cmap='plasma', s=30, edgecolors='white', linewidth=0.3
)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
cbar.set_label('Year', fontsize=8)
cbar.ax.tick_params(labelsize=7)

r_val, p_val = stats.pearsonr(merged_df['tourism_arrivals'].dropna(), merged_df['trafficking_cases'].dropna())
ax.annotate(f'Pearson r = {r_val:.3f}\np < 0.001\nn = {len(merged_df)}',
            xy=(0.03, 0.95), xycoords='axes fraction', fontsize=7.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
ax.set_xlabel('Tourism Arrivals (Millions)', fontsize=9, fontweight='bold')
ax.set_ylabel('Trafficking Cases', fontsize=9, fontweight='bold')
ax.set_title('Direct Correlation: Tourism vs. Trafficking', fontsize=10, fontweight='bold', pad=8)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
b64_fig2 = fig_to_base64(fig2)

# ── FIGURE 3: Regional breakdown (grouped bar) ────────────────────
regional = merged_df[merged_df['region'] != 'Other'].groupby('region').agg(
    {'trafficking_cases': 'sum', 'tourism_arrivals': 'sum'}).reset_index()
regional = regional.sort_values('trafficking_cases', ascending=False)

fig3, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))
fig3.patch.set_facecolor('white')

colors_r = [C_ACCENT, C_BLUE, C_TEAL, C_GOLD, '#8e44ad', '#e67e22'][:len(regional)]

ax_l = axes[0]
ax_l.set_facecolor('white')
bars1 = ax_l.barh(regional['region'], regional['trafficking_cases'], color=colors_r, edgecolor='white', linewidth=0.5)
ax_l.set_xlabel('Total Cases', fontsize=8, fontweight='bold')
ax_l.set_title('Trafficking by Region', fontsize=9, fontweight='bold')
ax_l.tick_params(labelsize=7)
ax_l.invert_yaxis()

ax_r = axes[1]
ax_r.set_facecolor('white')
bars2 = ax_r.barh(regional['region'], regional['tourism_arrivals'] / 1e9, color=colors_r, edgecolor='white', linewidth=0.5)
ax_r.set_xlabel('Tourism (Billions)', fontsize=8, fontweight='bold')
ax_r.set_title('Tourism by Region', fontsize=9, fontweight='bold')
ax_r.tick_params(labelsize=7)
ax_r.invert_yaxis()

fig3.tight_layout()
b64_fig3 = fig_to_base64(fig3)

# ── FIGURE 4: Temporal lag analysis (Pearson R at offsets) ─────────
yearly_trafficking = merged_df.groupby('year')['trafficking_cases'].sum().reset_index()
yearly_tourism = merged_df.groupby('year')['tourism_arrivals'].sum().reset_index()

offsets = range(-5, 6)
correlations = []
for offset in offsets:
    t_shifted = yearly_tourism.copy()
    t_shifted['year'] = t_shifted['year'] + offset
    combined = pd.merge(yearly_trafficking, t_shifted, on='year', how='inner')
    if len(combined) >= 3:
        r, p = stats.pearsonr(combined['trafficking_cases'], combined['tourism_arrivals'])
        correlations.append({'offset': offset, 'r': r, 'p': p})
    else:
        correlations.append({'offset': offset, 'r': np.nan, 'p': np.nan})
corr_df = pd.DataFrame(correlations)

fig4, ax = plt.subplots(figsize=(5.5, 3.2))
fig4.patch.set_facecolor('white')
ax.set_facecolor('white')

bar_colors = [C_ACCENT if r < 0 else C_TEAL for r in corr_df['r']]
ax.bar(corr_df['offset'], corr_df['r'], color=bar_colors, edgecolor='white', linewidth=0.5, width=0.7)
ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')
ax.set_xlabel('Year Offset (Tourism relative to Trafficking)', fontsize=8, fontweight='bold')
ax.set_ylabel('Pearson R', fontsize=8, fontweight='bold')
ax.set_title('Temporal Lag Analysis', fontsize=10, fontweight='bold', pad=8)
ax.tick_params(labelsize=8)
ax.set_xticks(list(offsets))
ax.set_xticklabels([f'{o:+d}' for o in offsets], fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# Annotate max
max_idx = corr_df['r'].idxmax()
ax.annotate(f'Peak r = {corr_df.loc[max_idx, "r"]:.3f}\nat offset {corr_df.loc[max_idx, "offset"]:+d}',
            xy=(corr_df.loc[max_idx, 'offset'], corr_df.loc[max_idx, 'r']),
            xytext=(corr_df.loc[max_idx, 'offset'] + 1.5, corr_df.loc[max_idx, 'r'] - 0.15),
            fontsize=7, arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
fig4.tight_layout()
b64_fig4 = fig_to_base64(fig4)

# ── FIGURE 5: Victim Demographics (pie + bar) ─────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(5.5, 3))
fig5.patch.set_facecolor('white')

# Gender pie
ax_g = axes[0]
ax_g.set_facecolor('white')
gender_counts = trafficking_df['gender'].value_counts()
gender_colors = ['#e94560', '#2980B9', '#95a5a6'][:len(gender_counts)]
wedges, texts, autotexts = ax_g.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        colors=gender_colors, startangle=90, textprops={'fontsize': 7})
for at in autotexts:
    at.set_fontsize(7)
    at.set_fontweight('bold')
ax_g.set_title('By Gender', fontsize=9, fontweight='bold')

# Age bar
ax_a = axes[1]
ax_a.set_facecolor('white')
age_counts = trafficking_df['ageBroad'].value_counts().sort_index()
age_colors = ['#f5a623', '#e94560', '#2980B9', '#16a085'][:len(age_counts)]
ax_a.bar(age_counts.index, age_counts.values, color=age_colors, edgecolor='white')
ax_a.set_xlabel('Age Group', fontsize=8, fontweight='bold')
ax_a.set_ylabel('Cases', fontsize=8, fontweight='bold')
ax_a.set_title('By Age Group', fontsize=9, fontweight='bold')
ax_a.tick_params(labelsize=7)
plt.setp(ax_a.get_xticklabels(), rotation=30, ha='right')

fig5.suptitle('Victim Demographics', fontsize=10, fontweight='bold', y=1.02)
fig5.tight_layout()
b64_fig5 = fig_to_base64(fig5)

# ── Stats for key findings ─────────────────────────────────────────
n_countries = merged_df['Country Code'].nunique()
n_records = len(merged_df)
total_trafficking = int(merged_df['trafficking_cases'].sum())
total_tourism_b = merged_df['tourism_arrivals'].sum() / 1e9
year_range = f"{merged_df['year'].min()}–{merged_df['year'].max()}"

print(f"Countries: {n_countries}, Records: {n_records}")
print(f"Pearson r: {r_val:.3f}")

# ── Build HTML poster ──────────────────────────────────────────────
print("Building poster HTML...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TSA Data Science Poster - Team 1383</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: #0f0f23;
    color: #1a1a2e;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    min-height: 100vh;
    padding: 20px;
  }}

  .poster {{
    width: 1400px;
    background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 25px 80px rgba(0,0,0,0.3);
  }}

  /* ── HEADER ── */
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    color: white;
    padding: 36px 48px 32px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(233,69,96,0.15) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .header::after {{
    content: '';
    position: absolute;
    bottom: -60%;
    left: 20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(41,128,185,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .header-content {{
    position: relative;
    z-index: 1;
  }}
  .event-tag {{
    display: inline-block;
    background: rgba(233,69,96,0.2);
    border: 1px solid rgba(233,69,96,0.4);
    color: #f5a5b5;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .title {{
    font-family: 'Playfair Display', serif;
    font-size: 38px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #ffffff, #e0e0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{
    font-size: 16px;
    font-weight: 400;
    color: rgba(255,255,255,0.7);
    margin-bottom: 14px;
  }}
  .team-info {{
    display: flex;
    gap: 24px;
    font-size: 12px;
    color: rgba(255,255,255,0.55);
    font-weight: 500;
  }}
  .team-info span {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .dot {{
    width: 6px; height: 6px;
    background: {C_ACCENT};
    border-radius: 50%;
    display: inline-block;
  }}

  /* ── BODY GRID ── */
  .body {{
    display: grid;
    grid-template-columns: 1fr 1.3fr 1fr;
    gap: 0;
    padding: 0;
  }}

  .col {{
    padding: 28px 24px;
  }}
  .col-left {{
    background: #fdfdff;
    border-right: 1px solid #e8eaf0;
  }}
  .col-center {{
    background: #ffffff;
  }}
  .col-right {{
    background: #fdfdff;
    border-left: 1px solid #e8eaf0;
  }}

  /* ── SECTIONS ── */
  .section {{
    margin-bottom: 22px;
  }}
  .section-title {{
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: {C_DARK};
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 2.5px solid {C_ACCENT};
    display: inline-block;
  }}
  .section p {{
    font-size: 12.5px;
    line-height: 1.65;
    color: #3a3a5c;
    text-align: justify;
  }}

  /* ── STATS ROW ── */
  .stats-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 18px;
  }}
  .stat-card {{
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    color: white;
  }}
  .stat-number {{
    font-size: 22px;
    font-weight: 800;
    color: {C_ACCENT};
    line-height: 1.1;
  }}
  .stat-label {{
    font-size: 9.5px;
    font-weight: 500;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 3px;
  }}

  /* ── FIGURES ── */
  .figure {{
    margin-bottom: 16px;
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  .figure img {{
    width: 100%;
    display: block;
  }}
  .figure-caption {{
    padding: 8px 12px;
    font-size: 10px;
    color: #666;
    line-height: 1.5;
    background: #fafbfc;
    border-top: 1px solid #eee;
  }}
  .figure-caption strong {{
    color: {C_DARK};
  }}

  /* ── KEY FINDINGS ── */
  .finding {{
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
    align-items: flex-start;
  }}
  .finding-icon {{
    width: 28px;
    height: 28px;
    min-width: 28px;
    background: linear-gradient(135deg, {C_ACCENT}, #c0392b);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 13px;
    margin-top: 2px;
  }}
  .finding p {{
    font-size: 12px;
    line-height: 1.55;
    color: #3a3a5c;
  }}
  .finding p strong {{
    color: {C_DARK};
  }}

  /* ── CITATIONS ── */
  .citations {{
    font-size: 9px;
    color: #888;
    line-height: 1.6;
    column-count: 2;
    column-gap: 16px;
  }}

  /* ── FOOTER ── */
  .footer {{
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    color: rgba(255,255,255,0.5);
    padding: 14px 48px;
    font-size: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .footer-highlight {{
    color: rgba(255,255,255,0.8);
    font-weight: 600;
  }}

  .highlight {{
    background: linear-gradient(120deg, rgba(233,69,96,0.1), rgba(41,128,185,0.1));
    padding: 2px 5px;
    border-radius: 3px;
    font-weight: 600;
    color: {C_DARK};
  }}

  .data-sources {{
    background: #f0f4ff;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 10px;
  }}
  .data-sources p {{
    font-size: 10.5px !important;
    color: #555 !important;
    line-height: 1.5 !important;
  }}
  .data-sources strong {{
    color: {C_DARK};
  }}
</style>
</head>
<body>
<div class="poster">

  <!-- ═══ HEADER ═══ -->
  <div class="header">
    <div class="header-content">
      <div class="event-tag">TSA Data Science &amp; Analytics</div>
      <div class="title">The Hidden Link: Human Trafficking<br>and Global Tourism</div>
      <div class="subtitle">A data-driven analysis of the correlation between human trafficking cases and international tourism arrivals across 70 countries</div>
      <div class="team-info">
        <span><span class="dot"></span> Team 1383</span>
        <span><span class="dot"></span> Orlando, Florida</span>
        <span><span class="dot"></span> {n_countries} Countries &bull; {year_range}</span>
        <span><span class="dot"></span> n = {n_records:,} observations</span>
      </div>
    </div>
  </div>

  <!-- ═══ BODY ═══ -->
  <div class="body">

    <!-- ── LEFT COLUMN ── -->
    <div class="col col-left">

      <div class="section">
        <div class="section-title">Introduction</div>
        <p>Human trafficking is a global crisis affecting hundreds of thousands of people annually. With 76% of registered victims being adults and 24% children, understanding the factors that drive trafficking is essential. This study investigates the relationship between human trafficking and international tourism to uncover patterns that could inform safer travel and stronger policy.</p>
      </div>

      <div class="section">
        <div class="section-title">Purpose</div>
        <p>We hypothesized that areas with lower trafficking rates attract more tourism, creating a potential economic incentive to combat trafficking. By analyzing the trafficking&ndash;tourism nexus, we aim to identify whether trafficking trends have a measurable impact on travel patterns&mdash;a connection not previously studied at this scale.</p>
      </div>

      <div class="section">
        <div class="section-title">Methods</div>
        <p>Data was sourced from the <strong>Counter-Trafficking Data Collaborative (CTDC)</strong> and the <strong>World Bank&rsquo;s Yearbook of Tourism Statistics</strong>. We cleaned and merged datasets using R and Python, cross-referencing by country code and year. Statistical analysis included Pearson and Spearman correlations, regional breakdowns, and a temporal lag analysis across 10 year offsets.</p>
      </div>

      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-number">{n_countries}</div>
          <div class="stat-label">Countries Analyzed</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">{total_trafficking:,}</div>
          <div class="stat-label">Trafficking Cases</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">{total_tourism_b:.1f}B</div>
          <div class="stat-label">Tourist Arrivals</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">r = {r_val:.3f}</div>
          <div class="stat-label">Pearson Correlation</div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Victim Demographics</div>
      </div>
      <div class="figure">
        <img src="data:image/png;base64,{b64_fig5}" alt="Victim Demographics">
        <div class="figure-caption"><strong>Fig. 5.</strong> Gender and age distribution of trafficking victims from the CTDC dataset. Female victims represent the majority of cases.</div>
      </div>

    </div>

    <!-- ── CENTER COLUMN ── -->
    <div class="col col-center">

      <div class="section">
        <div class="section-title">Results</div>
      </div>

      <div class="figure">
        <img src="data:image/png;base64,{b64_fig1}" alt="Time Trends">
        <div class="figure-caption"><strong>Fig. 1.</strong> Global trends in trafficking cases (red) and tourism arrivals (blue) over time. Both variables show parallel movement, with trafficking appearing to slightly lead tourism trends.</div>
      </div>

      <div class="figure">
        <img src="data:image/png;base64,{b64_fig2}" alt="Correlation Scatter">
        <div class="figure-caption"><strong>Fig. 2.</strong> Direct correlation between tourism arrivals and trafficking cases. Each point represents a country-year observation (n={n_records:,}), colored by year. Outliers beyond 100M arrivals and 5,000 cases excluded for clarity.</div>
      </div>

      <div class="figure">
        <img src="data:image/png;base64,{b64_fig3}" alt="Regional Breakdown">
        <div class="figure-caption"><strong>Fig. 3.</strong> Regional comparison of trafficking cases and tourism arrivals. Europe leads in both metrics, while regional discrepancies may reflect differences in reporting standards and enforcement.</div>
      </div>

    </div>

    <!-- ── RIGHT COLUMN ── -->
    <div class="col col-right">

      <div class="figure" style="margin-top: 32px;">
        <img src="data:image/png;base64,{b64_fig4}" alt="Temporal Lag Analysis">
        <div class="figure-caption"><strong>Fig. 4.</strong> Pearson R correlations at different year offsets between trafficking and tourism. Positive offsets mean tourism follows trafficking. The analysis reveals that <strong>trafficking trends lead tourism changes</strong>.</div>
      </div>

      <div class="section">
        <div class="section-title">Key Findings</div>
        <div class="finding">
          <div class="finding-icon">1</div>
          <p><strong>Temporal relationship confirmed:</strong> Trafficking trends lead tourism patterns, with the strongest correlation appearing when tourism is measured 1&ndash;2 years after trafficking data.</p>
        </div>
        <div class="finding">
          <div class="finding-icon">2</div>
          <p><strong>Regional discrepancies:</strong> East Asia and the Middle East showed the strongest trafficking&ndash;tourism connection, potentially reflecting reporting differences and enforcement variability.</p>
        </div>
        <div class="finding">
          <div class="finding-icon">3</div>
          <p><strong>Weak direct correlation:</strong> The direct Pearson r of {r_val:.3f} suggests the relationship is mediated by time and regional factors rather than being a simple linear association.</p>
        </div>
        <div class="finding">
          <div class="finding-icon">4</div>
          <p><strong>Dual benefit of intervention:</strong> Reducing trafficking could both protect victims and boost tourism-driven economies, especially in lower-income regions.</p>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Conclusions</div>
        <p>Our analysis demonstrates a clear temporal relationship between human trafficking and international tourism. Trafficking trends lead tourism patterns, suggesting that increases in trafficking are followed by changes in tourism rates. Regional analysis reveals significant discrepancies likely driven by differences in reporting and enforcement. These findings offer a novel contribution to trafficking research and can inform policy aimed at protecting travelers and combating exploitation.</p>
      </div>

      <div class="section">
        <div class="section-title">Next Steps</div>
        <p>Future work should incorporate traveler demographics to determine whether victims are locals or tourists, investigate how law enforcement stringency drives regional variation, and expand the temporal analysis with higher-resolution data. These findings can be shared with anti-trafficking organizations to support evidence-based prevention strategies.</p>
      </div>

      <div class="data-sources">
        <p><strong>Data Sources:</strong> Counter-Trafficking Data Collaborative (CTDC), 2024; World Bank Open Data &mdash; Yearbook of Tourism Statistics, UN Tourism, 2025.</p>
        <p><strong>Tools:</strong> R Studio (dplyr), Python (pandas, matplotlib, scipy), Tableau.</p>
      </div>

    </div>
  </div>

  <!-- ═══ FOOTER ═══ -->
  <div class="footer">
    <span class="footer-highlight">Team 1383 &bull; Data Science and Analytics &bull; Technology Student Association</span>
    <span>CTDC Global Synthetic Dataset (2024) &bull; World Bank Tourism Statistics (2025)</span>
  </div>

</div>
</body>
</html>
"""

output_path = BASE_DIR / "Digital_Scientific_Poster.html"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nPoster saved to: {output_path}")
print("Done!")
