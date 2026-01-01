"""
UI utility functions for the Streamlit app.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Auto-detect categorical and numeric columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with 'categorical' and 'numeric' column lists
    """
    categorical = []
    numeric = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Check if it's actually categorical (limited unique values)
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5 or df[col].nunique() < 20:
                categorical.append(col)
            else:
                # Might be string data, treat as categorical
                categorical.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (integer with few unique values)
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1 and df[col].nunique() < 10:
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            numeric.append(col)
    
    return {'categorical': categorical, 'numeric': numeric}


def validate_dataframes(ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Validate and align dataframes, handling mismatches.
    
    Args:
        ref_df: Reference DataFrame
        curr_df: Current DataFrame
        
    Returns:
        Tuple of (aligned_ref, aligned_curr, warnings)
    """
    warnings = []
    
    # Check for empty dataframes
    if ref_df.empty:
        warnings.append("⚠️ Reference dataframe is empty")
        return ref_df, curr_df, warnings
    
    if curr_df.empty:
        warnings.append("⚠️ Current dataframe is empty")
        return ref_df, curr_df, warnings
    
    # Check column compatibility
    ref_cols = set(ref_df.columns)
    curr_cols = set(curr_df.columns)
    
    common_cols = sorted(list(ref_cols & curr_cols))
    ref_only = ref_cols - curr_cols
    curr_only = curr_cols - ref_cols
    
    if ref_only:
        warnings.append(f"⚠️ Columns only in reference: {', '.join(ref_only)}")
    
    if curr_only:
        warnings.append(f"⚠️ Columns only in current: {', '.join(curr_only)}")
    
    if not common_cols:
        warnings.append("❌ No common columns found between datasets")
        return pd.DataFrame(), pd.DataFrame(), warnings
    
    # Use only common columns
    aligned_ref = ref_df[common_cols].copy()
    aligned_curr = curr_df[common_cols].copy()
    
    # Check for size mismatches
    if len(aligned_ref) != len(aligned_curr):
        warnings.append(f"⚠️ Row count mismatch: Reference={len(aligned_ref)}, Current={len(aligned_curr)}")
    
    return aligned_ref, aligned_curr, warnings


def process_large_file(uploaded_file, chunk_size: int = 10000) -> Optional[pd.DataFrame]:
    """
    Process large files in chunks.
    
    Args:
        uploaded_file: Uploaded file object
        chunk_size: Number of rows per chunk
        
    Returns:
        DataFrame or None if error
    """
    try:
        file_size = uploaded_file.size
        
        # If file is larger than 10MB, process in chunks
        if file_size > 10 * 1024 * 1024:  # 10MB
            st.info(f"📦 Large file detected ({file_size / (1024*1024):.2f} MB). Processing in chunks...")
            
            chunks = []
            chunk_iter = pd.read_csv(uploaded_file, chunksize=chunk_size)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunk_iter):
                chunks.append(chunk)
                progress = (i + 1) / 100  # Estimate based on chunks
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing chunk {i+1}...")
            
            progress_bar.empty()
            status_text.empty()
            
            df = pd.concat(chunks, ignore_index=True)
            st.success(f"✅ Loaded {len(df)} rows from large file")
            return df
        else:
            # Small file, load normally
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def get_shareable_link(params: Dict) -> str:
    """
    Generate shareable link with query parameters.
    
    Args:
        params: Dictionary of parameters to encode
        
    Returns:
        Shareable URL string
    """
    import urllib.parse
    
    try:
        # Try to get the current URL
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx and hasattr(ctx, 'session_id'):
            # Use a placeholder that users can replace
            base_url = "https://your-app.streamlit.app"  # User should replace this
        else:
            base_url = "http://localhost:8501"
    except:
        base_url = "https://your-app.streamlit.app"
    
    query_string = urllib.parse.urlencode(params)
    return f"{base_url}?{query_string}"


def apply_theme(theme: str):
    """
    Apply theme CSS.
    
    Args:
        theme: 'light' or 'dark'
    """
    if theme == 'dark':
        st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            .main .block-container {
                background-color: #0e1117;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #ffffff;
                color: #262730;
            }
            .main .block-container {
                background-color: #ffffff;
            }
            </style>
        """, unsafe_allow_html=True)

