"""
AI Drift Comparator - Enhanced Streamlit App
Compare data drift, model drift, and feature importance drift across multiple datasets.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
from datetime import datetime
import requests
from io import StringIO

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CRITICAL: Import sklearn patch FIRST to register 'max_error' scorer
# This must happen before any imports that use deepchecks or evidently
from utils._sklearn_patch import *  # noqa: F401, F403

from utils.drift_utils import (
    detect_drift_evidently,
    detect_drift_deepchecks,
    calculate_model_drift,
    get_top_drifting_features,
    compare_drift,
    compare_multiple_datasets,
    generate_report,
    calculate_feature_importance_drift,
    batch_model_comparison,
    export_drifted_features_csv
)
from utils.ui_utils import (
    detect_column_types,
    validate_dataframes,
    process_large_file,
    get_shareable_link,
    apply_theme
)
from data.generate_sample_data import (
    generate_classification_data,
    generate_drifted_data,
    load_adult_income_dataset,
    generate_demo_data
)

# Page configuration
st.set_page_config(
    page_title="AI Drift Comparator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .badge-red {
        background-color: #f44336;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .badge-yellow {
        background-color: #ff9800;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .badge-green {
        background-color: #4caf50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .drift-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .no-drift-alert {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_from_file(uploaded_file):
    """Load and cache data from uploaded file with large file support."""
    try:
        # Check file size
        file_size = uploaded_file.size
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return process_large_file(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


@st.cache_data
def generate_synthetic_reference(dataset_type, n_samples=1000):
    """Generate synthetic reference dataset."""
    if dataset_type == "Classification":
        return generate_classification_data(n_samples=n_samples, n_features=10)
    elif dataset_type == "Adult Income":
        return load_adult_income_dataset()
    return None


@st.cache_data
def generate_synthetic_current(reference_df, drift_percentage=0.2):
    """Generate synthetic current dataset with drift."""
    result = generate_drifted_data(reference_df, drift_percentage=drift_percentage)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, []


def get_drift_badge(drift_score, threshold_low, threshold_high):
    """Get badge color based on drift score."""
    if drift_score >= threshold_high:
        return '<span class="badge-red">HIGH DRIFT</span>'
    elif drift_score >= threshold_low:
        return '<span class="badge-yellow">MEDIUM DRIFT</span>'
    else:
        return '<span class="badge-green">LOW DRIFT</span>'


def send_webhook_alert(webhook_url, message, drift_score, threshold):
    """Send alert to webhook (Slack/Email)."""
    try:
        payload = {
            "text": f"🚨 Drift Alert: {message}",
            "drift_score": drift_score,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(webhook_url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Webhook error: {str(e)}")
        return False


def create_heatmap(df_drift_scores):
    """Create heatmap of drift scores across datasets."""
    fig = px.imshow(
        df_drift_scores.values,
        labels=dict(x="Dataset", y="Feature", color="Drift Score"),
        x=df_drift_scores.columns.tolist(),
        y=df_drift_scores.index.tolist(),
        aspect="auto",
        color_continuous_scale="Reds",
        title="Drift Scores Heatmap (per Feature per Dataset)"
    )
    fig.update_layout(height=max(400, len(df_drift_scores) * 20))
    return fig


def create_waterfall_chart(feature_drift_scores, top_n=20):
    """Create waterfall chart showing cumulative drift contribution."""
    top_features = get_top_drifting_features(feature_drift_scores, top_n)
    
    if not top_features:
        return None
    
    features = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]
    
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(features),
        x=features,
        y=scores,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Cumulative Drift Contribution (Waterfall)",
        showlegend=False,
        height=500
    )
    fig.update_xaxes(tickangle=-45)
    
    return fig


def create_time_series_drift(df, timestamp_col, feature_col, reference_period=None):
    """Create time-series drift visualization."""
    if timestamp_col not in df.columns:
        return None
    
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df_sorted = df.sort_values(timestamp_col)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted[timestamp_col],
            y=df_sorted[feature_col],
            mode='lines+markers',
            name=feature_col,
            line=dict(width=2)
        ))
        
        if reference_period:
            fig.add_vline(
                x=reference_period,
                line_dash="dash",
                line_color="red",
                annotation_text="Reference Period End"
            )
        
        fig.update_layout(
            title=f"Time-Series Drift: {feature_col}",
            xaxis_title="Time",
            yaxis_title=feature_col,
            height=400
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create time-series plot: {str(e)}")
        return None


def main():
    """Main application function."""
    # Handle query parameters for shareable links
    query_params = st.query_params
    
    # Theme toggle
    if 'theme' in query_params:
        theme = query_params['theme']
    else:
        theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], key='theme_select')
    
    apply_theme(theme.lower())
    
    # Header
    st.markdown('<h1 class="main-header">📊 AI Drift Comparator</h1>', unsafe_allow_html=True)
    st.markdown("Compare data drift, model drift, and feature importance drift across multiple datasets.")
    
    # Shareable link button
    col_link1, col_link2 = st.columns([3, 1])
    with col_link1:
        st.caption("💡 Tip: Use shareable links to share your analysis configuration")
    with col_link2:
        if st.button("🔗 Get Shareable Link"):
            share_params = {
                'theme': theme.lower(),
            }
            shareable_url = get_shareable_link(share_params)
            st.code(shareable_url, language=None)
            st.info("📋 Copy this link and replace 'your-app' with your Streamlit Cloud URL")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Detection method
        detection_method = st.selectbox(
            "Detection Method",
            ["Evidently AI", "Deepchecks", "Both"],
            help="Select drift detection library"
        )
        
        # Thresholds
        st.subheader("Thresholds")
        threshold = st.slider(
            "Drift Threshold",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Threshold for detecting drift"
        )
        
        threshold_low = st.slider(
            "Low Alert Threshold",
            min_value=0.01,
            max_value=0.3,
            value=0.05,
            step=0.01
        )
        
        threshold_high = st.slider(
            "High Alert Threshold",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.01
        )
        
        # Webhook configuration
        st.divider()
        st.subheader("🔔 Alert Configuration")
        enable_webhooks = st.checkbox("Enable Webhook Alerts", value=False)
        webhook_url = None
        if enable_webhooks:
            webhook_url = st.text_input("Webhook URL (Slack/Email)", placeholder="https://hooks.slack.com/...")
        
        # Data source
        st.divider()
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Choose data source",
            ["Upload CSV", "Synthetic Dataset"],
            help="Upload your own CSV files or use built-in synthetic datasets"
        )
        
        # Multi-dataset mode
        multi_dataset_mode = st.checkbox("Multi-Dataset Comparison Mode", value=False)
    
    # Data loading section
    st.header("📥 Data Loading")
    
    datasets = {}
    
    if data_source == "Upload CSV":
        if multi_dataset_mode:
            st.subheader("Upload Multiple Datasets")
            num_datasets = st.number_input("Number of datasets", min_value=2, max_value=10, value=3, step=1)
            
            for i in range(num_datasets):
                dataset_name = st.text_input(f"Dataset {i+1} name", value=f"dataset_{i+1}", key=f"name_{i}")
                uploaded_file = st.file_uploader(
                    f"Upload {dataset_name} CSV",
                    type=['csv'],
                    key=f'upload_{i}'
                )
                if uploaded_file is not None:
                    df = load_data_from_file(uploaded_file)
                    if df is not None and not df.empty:
                        datasets[dataset_name] = df
                        st.success(f"✅ {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
                    elif df is not None and df.empty:
                        st.warning(f"⚠️ {dataset_name} is empty")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Reference Data")
                ref_file = st.file_uploader("Upload reference CSV", type=['csv'], key='ref_upload')
                if ref_file is not None:
                    df_ref = load_data_from_file(ref_file)
                    if df_ref is not None and not df_ref.empty:
                        datasets['reference'] = df_ref
                        st.success(f"✅ Loaded {len(df_ref)} rows")
                    elif df_ref is not None and df_ref.empty:
                        st.warning("⚠️ Reference file is empty")
                    else:
                        st.error("❌ Failed to load reference file")
            
            with col2:
                st.subheader("Current Data")
                curr_file = st.file_uploader("Upload current CSV", type=['csv'], key='curr_upload')
                if curr_file is not None:
                    df_curr = load_data_from_file(curr_file)
                    if df_curr is not None and not df_curr.empty:
                        datasets['current'] = df_curr
                        st.success(f"✅ Loaded {len(df_curr)} rows")
                    elif df_curr is not None and df_curr.empty:
                        st.warning("⚠️ Current file is empty")
                    else:
                        st.error("❌ Failed to load current file")
    else:  # Synthetic Dataset
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_type = st.selectbox(
                "Dataset Type",
                ["Classification", "Adult Income", "Demo Data (Normal + Drift)"],
                key='dataset_type'
            )
            n_samples = st.slider("Number of Samples", min_value=500, max_value=5000, value=1000, step=500, key='n_samples')
            
            if st.button("Generate Reference Data", key='gen_ref'):
                with st.spinner("Generating reference dataset..."):
                    if dataset_type == "Demo Data (Normal + Drift)":
                        ref_data, curr_data, drifted_features = generate_demo_data(n_samples=n_samples)
                        datasets['reference'] = ref_data
                        datasets['current'] = curr_data
                        st.success(f"✅ Generated reference: {len(ref_data)} rows")
                        st.success(f"✅ Generated current: {len(curr_data)} rows (30% features drifted)")
                        if drifted_features:
                            st.info(f"Drifted features: {', '.join(drifted_features)}")
                    else:
                        datasets['reference'] = generate_synthetic_reference(dataset_type, n_samples)
                        st.success(f"✅ Generated {len(datasets['reference'])} rows")
        
        with col2:
            if dataset_type != "Demo Data (Normal + Drift)":
                drift_percentage = st.slider(
                    "Drift Percentage",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    key='drift_pct'
                )
                
                if st.button("Generate Current Data (Drifted)", key='gen_curr'):
                    if 'reference' in datasets:
                        with st.spinner("Generating drifted dataset..."):
                            curr_data, drifted_features = generate_synthetic_current(datasets['reference'], drift_percentage)
                            datasets['current'] = curr_data
                            st.success(f"✅ Generated {len(datasets['current'])} rows")
                            if drifted_features:
                                st.info(f"Drifted features: {', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}")
                    else:
                        st.warning("⚠️ Please generate reference data first")
    
    # Display data preview and validate
    if datasets:
        with st.expander("📋 Data Preview"):
            for name, df in datasets.items():
                if df is not None and not df.empty:
                    st.subheader(f"{name} ({df.shape[0]} rows × {df.shape[1]} cols)")
                    
                    # Show column types
                    col_types = detect_column_types(df)
                    if col_types['categorical']:
                        st.caption(f"Categorical: {', '.join(col_types['categorical'][:5])}{'...' if len(col_types['categorical']) > 5 else ''}")
                    if col_types['numeric']:
                        st.caption(f"Numeric: {', '.join(col_types['numeric'][:5])}{'...' if len(col_types['numeric']) > 5 else ''}")
                    
                    st.dataframe(df.head(5), use_container_width=True)
                else:
                    st.warning(f"⚠️ {name} is empty or invalid")
    
    # Main analysis tabs
    if len(datasets) >= 2:
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📊 Data Drift", "🤖 Model Drift", "🎯 Feature Importance Drift"])
        
        with tab1:
            analyze_data_drift_tab(datasets, detection_method, threshold, threshold_low, threshold_high, webhook_url if enable_webhooks else None)
        
        with tab2:
            analyze_model_drift_tab(datasets, threshold, threshold_low, threshold_high, webhook_url if enable_webhooks else None)
        
        with tab3:
            analyze_feature_importance_tab(datasets)
    else:
        st.info("👆 Please load or generate at least 2 datasets to begin analysis.")


def analyze_data_drift_tab(datasets, detection_method, threshold, threshold_low, threshold_high, webhook_url):
    """Data Drift Analysis Tab."""
    st.header("📊 Data Drift Analysis")
    
    # Validate datasets
    if 'reference' in datasets and 'current' in datasets:
        ref_data = datasets['reference']
        curr_data = datasets['current']
        
        # Validate and align dataframes
        aligned_ref, aligned_curr, warnings = validate_dataframes(ref_data, curr_data)
        
        # Display warnings
        for warning in warnings:
            st.warning(warning)
        
        if aligned_ref.empty or aligned_curr.empty:
            st.error("❌ Cannot perform drift analysis: datasets are empty or have no common columns")
            return
        
        ref_data = aligned_ref
        curr_data = aligned_curr
    
    if len(datasets) >= 3:
        # Multi-dataset comparison
        st.subheader("Multi-Dataset Comparison")
        
        reference_name = st.selectbox("Select Reference Dataset", list(datasets.keys()), key='ref_select')
        
        if st.button("Compare All Datasets", key='compare_all'):
            with st.spinner("Comparing datasets..."):
                method = 'evidently' if detection_method in ['Evidently AI', 'Both'] else 'deepchecks'
                df_comparison = compare_multiple_datasets(datasets, reference_name, method, threshold)
                
                if not df_comparison.empty:
                    st.subheader("📈 Drift Scores Comparison Table")
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    # Heatmap
                    st.subheader("🔥 Drift Scores Heatmap")
                    fig_heatmap = create_heatmap(df_comparison)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Export
                    csv_export = df_comparison.to_csv()
                    st.download_button(
                        label="📥 Download Comparison CSV",
                        data=csv_export,
                        file_name="drift_comparison.csv",
                        mime="text/csv"
                    )
    else:
        # Standard two-dataset comparison
        if 'reference' in datasets and 'current' in datasets:
            ref_data = datasets['reference']
            curr_data = datasets['current']
            
            # Check column compatibility
            ref_cols = set(ref_data.columns)
            curr_cols = set(curr_data.columns)
            
            if ref_cols != curr_cols:
                st.warning("⚠️ Column mismatch detected. Using common columns only.")
                common_cols = sorted(list(ref_cols & curr_cols))
                ref_data = ref_data[common_cols]
                curr_data = curr_data[common_cols]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            
            # Evidently AI detection
            if detection_method in ["Evidently AI", "Both"]:
                status_text.text("🔍 Running Evidently AI drift detection...")
                progress_bar.progress(10)
                
                try:
                    evidently_results = detect_drift_evidently(ref_data, curr_data, threshold)
                    results['evidently'] = evidently_results
                    progress_bar.progress(50)
                except Exception as e:
                    st.error(f"Error in Evidently AI detection: {str(e)}")
                    results['evidently'] = {'error': str(e)}
            
            # Deepchecks detection
            if detection_method in ["Deepchecks", "Both"]:
                status_text.text("🔬 Running Deepchecks drift detection...")
                progress_bar.progress(60)
                
                try:
                    target_col = None
                    if 'target' in curr_data.columns:
                        target_col = 'target'
                    
                    deepchecks_results = detect_drift_deepchecks(ref_data, curr_data, target_col)
                    results['deepchecks'] = deepchecks_results
                    progress_bar.progress(90)
                except Exception as e:
                    st.error(f"Error in Deepchecks detection: {str(e)}")
                    results['deepchecks'] = {'error': str(e)}
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            st.success("Drift analysis completed successfully!")
            
            # Clear progress after a moment
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if detection_method in ["Evidently AI", "Both"] and 'evidently' in results:
                display_evidently_results(
                    results['evidently'], threshold, threshold_low, threshold_high,
                    ref_data, curr_data, webhook_url
                )
            
            if detection_method in ["Deepchecks", "Both"] and 'deepchecks' in results:
                display_deepchecks_results(results['deepchecks'])
            
            # Export section
            st.divider()
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON export
                if results:
                    export_json = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="📥 Download JSON",
                        data=export_json,
                        file_name="drift_results.json",
                        mime="application/json"
                    )
            
            with col2:
                # CSV export (drifted features)
                if 'evidently' in results and 'feature_drift_scores' in results['evidently']:
                    df_drifted = export_drifted_features_csv(
                        results['evidently']['feature_drift_scores'], threshold
                    )
                    if not df_drifted.empty:
                        csv_data = df_drifted.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Drifted Features CSV",
                            data=csv_data,
                            file_name="drifted_features.csv",
                            mime="text/csv"
                        )
            
            with col3:
                # HTML report (Evidently)
                if 'evidently' in results and 'full_report' in results['evidently']:
                    try:
                        from evidently.report import Report
                        from evidently.metrics import DatasetDriftMetric, DataDriftTable
                        
                        report = Report(metrics=[DatasetDriftMetric(threshold=threshold), DataDriftTable()])
                        report.run(reference_data=ref_data, current_data=curr_data)
                        html_report = report.get_html()
                        
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=html_report,
                            file_name="drift_report.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.warning(f"Could not generate HTML report: {str(e)}")


def display_evidently_results(results, threshold, threshold_low, threshold_high, ref_data, curr_data, webhook_url):
    """Display Evidently AI results with enhanced visualizations."""
    st.subheader("📊 Evidently AI Results")
    
    if 'error' in results:
        st.error(f"Error: {results['error']}")
        return
    
    # Key metrics with badges
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dataset_drifted = results.get('dataset_drifted', False)
        st.metric("Dataset Drifted", "Yes" if dataset_drifted else "No")
        if dataset_drifted:
            st.markdown(get_drift_badge(1.0, threshold_low, threshold_high), unsafe_allow_html=True)
    
    with col2:
        num_drifted = results.get('number_of_drifted_features', 0)
        st.metric("Drifted Features", num_drifted)
    
    with col3:
        share_drifted = results.get('share_of_drifted_features', 0.0)
        st.metric("Share of Drifted Features", f"{share_drifted:.2%}")
        st.markdown(get_drift_badge(share_drifted, threshold_low, threshold_high), unsafe_allow_html=True)
    
    with col4:
        total_features = len(results.get('feature_drift_scores', {}))
        st.metric("Total Features", total_features)
    
    # Alert if drift detected
    max_drift_score = 0.0
    if results.get('feature_drift_scores'):
        max_drift_score = max(
            (v.get('drift_score', 0.0) if isinstance(v, dict) else float(v))
            for v in results['feature_drift_scores'].values()
        )
    
    if max_drift_score >= threshold_high:
        st.error(f"🚨 HIGH DRIFT DETECTED! Maximum drift score: {max_drift_score:.4f}")
        if webhook_url:
            send_webhook_alert(webhook_url, f"High drift detected: {max_drift_score:.4f}", max_drift_score, threshold_high)
    elif max_drift_score >= threshold_low:
        st.warning(f"⚠️ MEDIUM DRIFT DETECTED! Maximum drift score: {max_drift_score:.4f}")
        if webhook_url:
            send_webhook_alert(webhook_url, f"Medium drift detected: {max_drift_score:.4f}", max_drift_score, threshold_low)
    elif results.get('dataset_drifted', False):
        st.info(f"✅ Low drift detected. Maximum drift score: {max_drift_score:.4f}")
    
    # Feature drift scores
    feature_scores = results.get('feature_drift_scores', {})
    if feature_scores:
        st.subheader("📈 Feature Drift Scores")
        
        # Top drifting features
        top_features = get_top_drifting_features(feature_scores, top_n=20)
        
        if top_features:
            # Create DataFrame
            df_drift = pd.DataFrame(top_features, columns=['Feature', 'Drift Score'])
            
            # Bar chart
            fig = px.bar(
                df_drift,
                x='Drift Score',
                y='Feature',
                orientation='h',
                title="Top Drifting Features (by Drift Score)",
                color='Drift Score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Waterfall chart
            st.subheader("💧 Cumulative Drift Contribution (Waterfall)")
            fig_waterfall = create_waterfall_chart(feature_scores, top_n=20)
            if fig_waterfall:
                st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # Table
            st.dataframe(df_drift, use_container_width=True)
            
            # Histogram comparison for top drifting feature
            if len(top_features) > 0:
                top_feature_name = top_features[0][0]
                st.subheader(f"📊 Distribution Comparison: {top_feature_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_ref = px.histogram(
                        ref_data[top_feature_name],
                        title=f"Reference: {top_feature_name}",
                        nbins=30
                    )
                    st.plotly_chart(fig_ref, use_container_width=True)
                
                with col2:
                    fig_curr = px.histogram(
                        curr_data[top_feature_name],
                        title=f"Current: {top_feature_name}",
                        nbins=30,
                        color_discrete_sequence=['red']
                    )
                    st.plotly_chart(fig_curr, use_container_width=True)
                
                # Overlay comparison
                fig_overlay = go.Figure()
                fig_overlay.add_trace(go.Histogram(
                    x=ref_data[top_feature_name],
                    name='Reference',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig_overlay.add_trace(go.Histogram(
                    x=curr_data[top_feature_name],
                    name='Current',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig_overlay.update_layout(
                    title=f"Overlay Comparison: {top_feature_name}",
                    xaxis_title=top_feature_name,
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                st.plotly_chart(fig_overlay, use_container_width=True)
                
                # Time-series if timestamp column exists
                timestamp_cols = [col for col in ref_data.columns if 'time' in col.lower() or 'date' in col.lower()]
                if timestamp_cols:
                    timestamp_col = st.selectbox("Select Timestamp Column", timestamp_cols, key='ts_col')
                    if timestamp_col:
                        fig_ts = create_time_series_drift(
                            pd.concat([ref_data, curr_data], ignore_index=True),
                            timestamp_col,
                            top_feature_name
                        )
                        if fig_ts:
                            st.plotly_chart(fig_ts, use_container_width=True)


def display_deepchecks_results(results):
    """Display Deepchecks results."""
    st.subheader("🔬 Deepchecks Results (Validation)")
    
    if 'error' in results:
        st.error(f"Error: {results['error']}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Drift Detected", "Yes" if results.get('drift_detected', False) else "No")
    
    with col2:
        drift_score = results.get('drift_score', 0.0)
        st.metric("Overall Drift Score", f"{drift_score:.4f}")
    
    # Feature-level scores
    feature_scores = results.get('feature_drift_scores', {})
    if feature_scores:
        df_deepchecks = pd.DataFrame(
            list(feature_scores.items()),
            columns=['Feature', 'Drift Score']
        )
        df_deepchecks = df_deepchecks.sort_values('Drift Score', ascending=False).head(10)
        
        fig = px.bar(
            df_deepchecks,
            x='Drift Score',
            y='Feature',
            orientation='h',
            title="Top Drifting Features (Deepchecks)",
            color='Drift Score',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def analyze_model_drift_tab(datasets, threshold, threshold_low, threshold_high, webhook_url):
    """Model Drift Analysis Tab."""
    st.header("🤖 Model Drift Analysis")
    
    # Batch mode toggle
    batch_mode = st.checkbox("Batch Mode: Compare Multiple Models", value=False)
    
    if 'reference' not in datasets or 'current' not in datasets:
        st.warning("⚠️ Please load reference and current datasets for model drift analysis.")
        return
    
    ref_data = datasets['reference']
    curr_data = datasets['current']
    
    # Check for target column
    target_col = None
    if 'target' in ref_data.columns:
        target_col = 'target'
    else:
        st.warning("⚠️ No 'target' column found. Please ensure your data has a target column.")
        return
    
    if target_col not in curr_data.columns:
        st.error("❌ Target column not found in current data.")
        return
    
    if batch_mode:
        st.subheader("Batch Model Comparison")
        selected_models = st.multiselect(
            "Select Models to Compare",
            ["RandomForest", "XGBoost", "LogisticRegression"],
            default=["RandomForest", "XGBoost", "LogisticRegression"]
        )
        
        if st.button("Run Batch Comparison", key='batch_compare'):
            with st.spinner("Training and comparing models..."):
                batch_results = batch_model_comparison(ref_data, curr_data, target_col, selected_models)
                
                # Display results in table
                results_data = []
                for model_name, result in batch_results.items():
                    if 'error' not in result:
                        ref_metrics = result.get('reference_metrics', {})
                        curr_metrics = result.get('current_metrics', {})
                        perf_drop = result.get('performance_drop', 0.0)
                        is_clf = result.get('is_classification', True)
                        
                        if is_clf:
                            results_data.append({
                                'Model': model_name,
                                'Ref Accuracy': f"{ref_metrics.get('accuracy', 0.0):.4f}",
                                'Curr Accuracy': f"{curr_metrics.get('accuracy', 0.0):.4f}",
                                'Performance Drop': f"{perf_drop:.4f}",
                                'Ref F1': f"{ref_metrics.get('f1', 0.0):.4f}" if ref_metrics.get('f1') else "N/A"
                            })
                        else:
                            results_data.append({
                                'Model': model_name,
                                'Ref R²': f"{ref_metrics.get('r2', 0.0):.4f}",
                                'Curr R²': f"{curr_metrics.get('r2', 0.0):.4f}",
                                'Performance Drop (MSE)': f"{perf_drop:.4f}",
                                'Ref MSE': f"{ref_metrics.get('mse', 0.0):.4f}"
                            })
                    else:
                        results_data.append({
                            'Model': model_name,
                            'Error': result.get('error', 'Unknown error')
                        })
                
                if results_data:
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Visualization
                    if len(results_data) > 0 and 'Performance Drop' in results_data[0]:
                        fig = px.bar(
                            df_results,
                            x='Model',
                            y='Performance Drop',
                            title="Model Performance Drop Comparison",
                            color='Performance Drop',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        # Single model analysis
        with st.spinner("Training model and calculating performance..."):
            model_results = calculate_model_drift(ref_data, curr_data, target_col)
        
        if 'error' in model_results:
            st.error(f"Error: {model_results['error']}")
            return
        
        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        
        ref_metrics = model_results.get('reference_metrics', {})
        curr_metrics = model_results.get('current_metrics', {})
        performance_drop = model_results.get('performance_drop', 0.0)
        is_classification = model_results.get('is_classification', True)
        
        if is_classification:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Reference Accuracy", f"{ref_metrics.get('accuracy', 0.0):.4f}")
            
            with col2:
                st.metric("Current Accuracy", f"{curr_metrics.get('accuracy', 0.0):.4f}")
            
            with col3:
                delta_color = "inverse" if performance_drop > 0 else "normal"
                st.metric("Performance Drop", f"{performance_drop:.4f}", delta=f"{performance_drop:.4f}", delta_color=delta_color)
                if abs(performance_drop) >= threshold_high:
                    st.markdown(get_drift_badge(abs(performance_drop), threshold_low, threshold_high), unsafe_allow_html=True)
                    if webhook_url:
                        send_webhook_alert(webhook_url, f"High model performance drop: {performance_drop:.4f}", abs(performance_drop), threshold_high)
            
            # Additional metrics
            col4, col5, col6, col7 = st.columns(4)
            with col4:
                st.metric("Reference Precision", f"{ref_metrics.get('precision', 0.0):.4f}" if ref_metrics.get('precision') is not None else "N/A")
            with col5:
                st.metric("Current Precision", f"{curr_metrics.get('precision', 0.0):.4f}" if curr_metrics.get('precision') is not None else "N/A")
            with col6:
                st.metric("Reference Recall", f"{ref_metrics.get('recall', 0.0):.4f}" if ref_metrics.get('recall') is not None else "N/A")
            with col7:
                st.metric("Reference F1", f"{ref_metrics.get('f1', 0.0):.4f}" if ref_metrics.get('f1') is not None else "N/A")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Reference MSE", f"{ref_metrics.get('mse', 0.0):.4f}")
            with col2:
                st.metric("Current MSE", f"{curr_metrics.get('mse', 0.0):.4f}")
            with col3:
                st.metric("Reference R²", f"{ref_metrics.get('r2', 0.0):.4f}")
            with col4:
                st.metric("Current R²", f"{curr_metrics.get('r2', 0.0):.4f}")
        
        # Performance comparison chart
        if is_classification:
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy'],
                'Reference': [ref_metrics.get('accuracy', 0.0)],
                'Current': [curr_metrics.get('accuracy', 0.0)]
            })
        else:
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score'],
                'Reference': [ref_metrics.get('r2', 0.0)],
                'Current': [curr_metrics.get('r2', 0.0)]
            })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics_df['Metric'], y=metrics_df['Reference'], name='Reference', marker_color='blue'))
        fig.add_trace(go.Bar(x=metrics_df['Metric'], y=metrics_df['Current'], name='Current', marker_color='red'))
        fig.update_layout(title="Model Performance Comparison", yaxis_title="Score", barmode='group')
        st.plotly_chart(fig, use_container_width=True)


def analyze_feature_importance_tab(datasets):
    """Feature Importance Drift Analysis Tab."""
    st.header("🎯 Feature Importance Drift Analysis")
    
    if 'reference' not in datasets or 'current' not in datasets:
        st.warning("⚠️ Please load reference and current datasets for feature importance analysis.")
        return
    
    ref_data = datasets['reference']
    curr_data = datasets['current']
    
    # Check for target column
    target_col = None
    if 'target' in ref_data.columns:
        target_col = 'target'
    else:
        st.warning("⚠️ No 'target' column found.")
        return
    
    selected_models = st.multiselect(
        "Select Models for Feature Importance",
        ["RandomForest", "XGBoost", "LogisticRegression"],
        default=["RandomForest"]
    )
    
    if st.button("Analyze Feature Importance Drift", key='analyze_importance'):
        with st.spinner("Calculating feature importance drift..."):
            importance_results = calculate_feature_importance_drift(
                ref_data, curr_data, target_col, selected_models
            )
        
        if 'error' in importance_results:
            st.error(f"Error: {importance_results['error']}")
            return
        
        for model_name, result in importance_results.items():
            if 'error' in result:
                st.warning(f"{model_name}: {result['error']}")
                continue
            
            st.subheader(f"📊 {model_name} Feature Importance")
            
            ref_importance = result.get('reference_importance', {})
            curr_importance = result.get('current_importance', {})
            importance_drift = result.get('importance_drift', {})
            
            if importance_drift:
                # Create comparison DataFrame
                df_importance = pd.DataFrame({
                    'Feature': list(importance_drift.keys()),
                    'Reference Importance': [ref_importance.get(f, 0.0) for f in importance_drift.keys()],
                    'Current Importance': [curr_importance.get(f, 0.0) for f in importance_drift.keys()],
                    'Importance Drift': list(importance_drift.values())
                })
                df_importance = df_importance.sort_values('Importance Drift', ascending=False)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_importance['Feature'],
                    y=df_importance['Reference Importance'],
                    name='Reference',
                    marker_color='blue'
                ))
                fig.add_trace(go.Bar(
                    x=df_importance['Feature'],
                    y=df_importance['Current Importance'],
                    name='Current',
                    marker_color='red'
                ))
                fig.update_layout(
                    title=f"{model_name} Feature Importance Comparison",
                    xaxis_title="Feature",
                    yaxis_title="Importance",
                    barmode='group',
                    height=500
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Drift scores
                fig_drift = px.bar(
                    df_importance,
                    x='Feature',
                    y='Importance Drift',
                    title=f"{model_name} Feature Importance Drift",
                    color='Importance Drift',
                    color_continuous_scale='Reds'
                )
                fig_drift.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_drift, use_container_width=True)
                
                # Table
                st.dataframe(df_importance, use_container_width=True)


if __name__ == "__main__":
    main()

