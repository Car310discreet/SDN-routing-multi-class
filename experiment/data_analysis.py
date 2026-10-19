#!/usr/bin/env python3
"""
Data Profiling and Exploratory Data Analysis (EDA) Script for result_data.csv
Performs comprehensive dataset profiling using pandas-profiling / ydata-profiling
(or a built-in full-featured HTML & console profiling engine if running on Python environments
where ydata-profiling is not yet natively packaged).
"""

import os
import sys
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
# Set backend to Agg for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'


def load_dataset(csv_path):
    """Loads and validates the result_data.csv dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Successfully loaded dataset: {csv_path}")
    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def generate_ydata_profiling_report(df, output_html):
    """Attempts to generate a profile report using ydata_profiling or pandas_profiling."""
    try:
        from ydata_profiling import ProfileReport
        print("[INFO] Using ydata-profiling to generate report...")
        profile = ProfileReport(df, title="NSFNET Traffic Classification - Dataset Profile", explorative=True)
        profile.to_file(output_html)
        print(f"[SUCCESS] ydata-profiling report saved to: {output_html}")
        return True
    except ImportError:
        try:
            from pandas_profiling import ProfileReport
            print("[INFO] Using legacy pandas-profiling to generate report...")
            profile = ProfileReport(df, title="NSFNET Traffic Classification - Dataset Profile", explorative=True)
            profile.to_file(output_html)
            print(f"[SUCCESS] pandas-profiling report saved to: {output_html}")
            return True
        except Exception as e:
            print(f"[DEBUG] ProfileReport import/run error: {e}")
            return False


def generate_comprehensive_eda_report(df, output_html):
    """
    Generates a rich, interactive, standalone HTML profiling report
    with full statistical summaries, missingness analysis, correlation heatmaps,
    and distribution visualizations per traffic classification type.
    """
    print("[INFO] Generating comprehensive HTML profiling report...")

    # Basic statistics
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()
    memory_usage_kb = df.memory_usage(deep=True).sum() / 1024

    # Target class distribution
    target_col = 'traffic_type_numeric' if 'traffic_type_numeric' in df.columns else None
    target_counts = df[target_col].value_counts().sort_index() if target_col else pd.Series()
    target_pct = (target_counts / n_rows) * 100 if target_col else pd.Series()

    # Numeric summary
    desc = df.describe().T
    desc['median'] = df.median(numeric_only=True)
    desc['skew'] = df.skew(numeric_only=True)
    desc['kurtosis'] = df.kurtosis(numeric_only=True)
    desc['zeros'] = (df == 0).sum()
    desc['zeros_pct'] = (desc['zeros'] / n_rows) * 100

    # Correlation Matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method='pearson')

    # -------------------------------------------------------------
    # Generate Visualizations (Encoded as Base64 images in HTML)
    # -------------------------------------------------------------
    images = {}

    # 1. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(14, 11), dpi=120)
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)
    ticks = range(len(corr_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_matrix.columns, fontsize=8)
    ax.set_title("Pearson Correlation Heatmap", pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    images['corr_heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 2. Target Class Distribution Bar Chart
    if target_col:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        bars = ax.bar([f"Class {k}" for k in target_counts.index], target_counts.values,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='#222222', width=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + n_rows*0.01, f"{int(yval)} ({yval/n_rows*100:.1f}%)",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title("Traffic Type Target Distribution", fontsize=12, fontweight='bold')
        ax.set_ylabel("Sample Count")
        ax.set_ylim(0, max(target_counts.values) * 1.15)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        images['target_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 3. Selected Feature Distributions & Boxplots
    key_features = ['average_rtt', 'packet_loss', 'bits_per_second', 'retransmits',
                    'bidirectional_duration_ms', 'cpu_host_total']
    key_features = [f for f in key_features if f in df.columns]

    if key_features:
        fig, axes = plt.subplots(len(key_features), 2, figsize=(14, 3.2 * len(key_features)), dpi=110)
        for i, feat in enumerate(key_features):
            # Histogram + KDE
            ax_hist = axes[i, 0] if len(key_features) > 1 else axes[0]
            ax_hist.hist(df[feat].dropna(), bins=30, color='#3498db', edgecolor='#2980b9', alpha=0.7)
            ax_hist.set_title(f"Distribution: {feat}", fontsize=10, fontweight='bold')
            ax_hist.set_ylabel("Frequency")

            # Boxplot by Target Class
            ax_box = axes[i, 1] if len(key_features) > 1 else axes[1]
            if target_col:
                grouped = [df[df[target_col] == c][feat].dropna().values for c in sorted(df[target_col].unique())]
                ax_box.boxplot(grouped, tick_labels=[f"Class {c}" for c in sorted(df[target_col].unique())], patch_artist=True,
                               boxprops=dict(facecolor='#a8d8ea', color='#2c3e50'))
                ax_box.set_title(f"{feat} by Traffic Class", fontsize=10, fontweight='bold')
            else:
                ax_box.boxplot(df[feat].dropna())
                ax_box.set_title(f"Boxplot: {feat}", fontsize=10, fontweight='bold')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        images['feature_distributions'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    # -------------------------------------------------------------
    # Build HTML Content
    # -------------------------------------------------------------
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Profile Report - result_data.csv</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f4f6f9;
            color: #2d3748;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
            background: #ffffff;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        }}
        h1, h2, h3 {{
            color: #1a202c;
            border-bottom: 2px solid #edf2f7;
            padding-bottom: 8px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }}
        .stat-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .stat-card .val {{
            font-size: 24px;
            font-weight: bold;
            color: #2b6cb0;
            margin-top: 6px;
        }}
        .stat-card .lbl {{
            font-size: 13px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f7fafc;
            color: #4a5568;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .img-container {{
            text-align: center;
            margin: 24px 0;
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        .section-box {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin: 24px 0;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-info {{ background: #ebf8ff; color: #2b6cb0; }}
        .badge-success {{ background: #f0fff4; color: #276749; }}
        .badge-warn {{ background: #fffaf0; color: #9c4221; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Dataset Profile Report: <code>result_data.csv</code></h1>
        <p style="color: #718096;">Automated Exploratory Data Analysis & Feature Profiling for AI-Based Routing and Traffic Classification in SDN</p>

        <!-- Key Metrics Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="lbl">Total Observations</div>
                <div class="val">{n_rows:,}</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Total Features</div>
                <div class="val">{n_cols}</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Missing Values</div>
                <div class="val">{missing_cells} ({missing_pct:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Duplicate Rows</div>
                <div class="val">{duplicate_rows}</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Memory Footprint</div>
                <div class="val">{memory_usage_kb:.1f} KB</div>
            </div>
        </div>

        <!-- Target Class Breakdown -->
        <div class="section-box">
            <h2>🎯 Target Variable: <code>traffic_type_numeric</code></h2>
            <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 20px;">
                <div style="flex: 1; min-width: 280px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Count</th>
                                <th>Proportion</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f"<tr><td><strong>Class {k}</strong></td><td>{target_counts[k]}</td><td>{target_pct[k]:.2f}%</td></tr>" for k in target_counts.index])}
                        </tbody>
                    </table>
                </div>
                <div style="flex: 1; min-width: 320px; text-align: center;">
                    <img src="data:image/png;base64,{images.get('target_dist', '')}" alt="Target Distribution" style="max-width: 100%;">
                </div>
            </div>
        </div>

        <!-- Detailed Variable Summary Table -->
        <div class="section-box">
            <h2>📈 Variable Statistics & Summary</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Mean</th>
                            <th>Std Dev</th>
                            <th>Min</th>
                            <th>25%</th>
                            <th>Median</th>
                            <th>75%</th>
                            <th>Max</th>
                            <th>Skewness</th>
                            <th>Zeroes (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([
                            f"<tr>"
                            f"<td><strong>{feat}</strong></td>"
                            f"<td>{desc.loc[feat, 'mean']:.2f}</td>"
                            f"<td>{desc.loc[feat, 'std']:.2f}</td>"
                            f"<td>{desc.loc[feat, 'min']:.2f}</td>"
                            f"<td>{desc.loc[feat, '25%']:.2f}</td>"
                            f"<td>{desc.loc[feat, 'median']:.2f}</td>"
                            f"<td>{desc.loc[feat, '75%']:.2f}</td>"
                            f"<td>{desc.loc[feat, 'max']:.2f}</td>"
                            f"<td>{desc.loc[feat, 'skew']:.2f}</td>"
                            f"<td>{desc.loc[feat, 'zeros_pct']:.1f}%</td>"
                            f"</tr>"
                            for feat in desc.index
                        ])}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Correlation Heatmap -->
        <div class="section-box">
            <h2>🔥 Feature Correlation Analysis</h2>
            <p style="color: #718096;">Pearson correlation matrix across all QoS and traffic classification metrics.</p>
            <div class="img-container">
                <img src="data:image/png;base64,{images.get('corr_heatmap', '')}" alt="Correlation Heatmap">
            </div>
        </div>

        <!-- Key Feature Distributions -->
        <div class="section-box">
            <h2>📊 Key Feature Distributions & Class Variations</h2>
            <p style="color: #718096;">Histograms and Boxplot distributions segmented across target traffic classes.</p>
            <div class="img-container">
                <img src="data:image/png;base64,{images.get('feature_distributions', '')}" alt="Feature Distributions">
            </div>
        </div>

        <!-- Sample Data Preview -->
        <div class="section-box">
            <h2>👀 First 5 Observations</h2>
            <div style="overflow-x: auto;">
                {df.head().to_html(classes="table", border=0, justify="left")}
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[SUCCESS] Comprehensive HTML profiling report generated at: {output_html}")


def print_console_summary(df):
    """Prints a concise statistical summary to the terminal."""
    print("=" * 70)
    print("           RESULT_DATA.CSV SUMMARY OVERVIEW")
    print("=" * 70)
    print(f"Total Rows:     {df.shape[0]}")
    print(f"Total Columns:  {df.shape[1]}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicates:     {df.duplicated().sum()}")
    print("-" * 70)
    if 'traffic_type_numeric' in df.columns:
        print("Target Class Breakdown (`traffic_type_numeric`):")
        counts = df['traffic_type_numeric'].value_counts().sort_index()
        for c, count in counts.items():
            print(f"  - Class {c}: {count} samples ({count/len(df)*100:.2f}%)")
    print("=" * 70)


def main():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(exp_dir, "result_data.csv")
    output_html = os.path.join(exp_dir, "result_data_profile_report.html")

    df = load_dataset(csv_path)
    print_console_summary(df)

    # First attempt native ydata_profiling / pandas_profiling
    success = generate_ydata_profiling_report(df, output_html)

    # If ydata_profiling is not installed, use full standalone HTML EDA profiler
    if not success:
        generate_comprehensive_eda_report(df, output_html)


if __name__ == "__main__":
    main()
