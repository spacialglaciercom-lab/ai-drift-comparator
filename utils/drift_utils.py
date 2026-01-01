"""
Drift detection utilities using Evidently AI and Deepchecks.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDrift
import json

# Try to import Evidently with fallback for different versions
HAS_EVIDENTLY = False
HAS_EVIDENTLY_REPORT = False
HAS_DATA_DRIFT_TABLE = False
HAS_DATASET_DRIFT_METRIC = False
HAS_TEST_NUMBER_OF_DRIFTED = False
Report = None
TestSuite = None
DataDriftTable = None
ColumnDriftMetric = None
DatasetDriftMetric = None
TestNumberOfDriftedFeatures = None

# Try importing Evidently Report (new API)
try:
    from evidently.report import Report
    HAS_EVIDENTLY_REPORT = True
    HAS_EVIDENTLY = True
except ImportError:
    # Try old API
    try:
        from evidently.dashboard import Dashboard
        from evidently.tabs import DataDriftTab
        HAS_EVIDENTLY = True
    except ImportError:
        pass

# Try to import Evidently metrics with fallback for different versions
if HAS_EVIDENTLY:
    try:
        from evidently.metrics import DataDriftTable, ColumnDriftMetric
        HAS_DATA_DRIFT_TABLE = True
    except ImportError:
        try:
            # Try alternative import paths
            from evidently.metric_preset import DataDriftPreset
            HAS_DATA_DRIFT_TABLE = True
        except ImportError:
            pass

    try:
        from evidently.metrics import DatasetDriftMetric
        HAS_DATASET_DRIFT_METRIC = True
    except ImportError:
        pass

    try:
        from evidently.test_suite import TestSuite
    except ImportError:
        pass

    try:
        from evidently.tests import TestNumberOfDriftedFeatures
        HAS_TEST_NUMBER_OF_DRIFTED = True
    except ImportError:
        pass


def _detect_drift_statistical(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
) -> Dict:
    """
    Fallback drift detection using statistical tests (KS test, etc.)
    when Evidently is not available.
    """
    from scipy import stats
    
    feature_drift_scores = {}
    number_of_drifted_features = 0
    dataset_drifted = False
    
    # Get common columns
    common_cols = set(reference_data.columns) & set(current_data.columns)
    
    for col in common_cols:
        ref_col = reference_data[col].dropna()
        curr_col = current_data[col].dropna()
        
        if len(ref_col) == 0 or len(curr_col) == 0:
            continue
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(ref_col):
            try:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(ref_col, curr_col)
                drift_score = 1 - p_value  # Convert p-value to drift score
                drift_detected = p_value < threshold
                
                feature_drift_scores[col] = {
                    'drift_score': drift_score,
                    'drift_detected': drift_detected,
                    'stat_test': 'KS_test',
                    'p_value': p_value
                }
                
                if drift_detected:
                    number_of_drifted_features += 1
                    dataset_drifted = True
            except Exception:
                feature_drift_scores[col] = {
                    'drift_score': 0.0,
                    'drift_detected': False,
                    'stat_test': 'Error'
                }
        else:
            # For categorical, use chi-square test
            try:
                ref_counts = ref_col.value_counts()
                curr_counts = curr_col.value_counts()
                
                # Align categories
                all_cats = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_cats]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_cats]
                
                if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                    chi2, p_value = stats.chisquare(curr_aligned, f_exp=ref_aligned)
                    drift_score = 1 - min(p_value, 1.0)
                    drift_detected = p_value < threshold
                    
                    feature_drift_scores[col] = {
                        'drift_score': drift_score,
                        'drift_detected': drift_detected,
                        'stat_test': 'Chi-square',
                        'p_value': p_value
                    }
                    
                    if drift_detected:
                        number_of_drifted_features += 1
                        dataset_drifted = True
            except Exception:
                feature_drift_scores[col] = {
                    'drift_score': 0.0,
                    'drift_detected': False,
                    'stat_test': 'Error'
                }
    
    total_features = len(feature_drift_scores) if feature_drift_scores else 1
    share_of_drifted_features = number_of_drifted_features / total_features if total_features > 0 else 0.0
    
    return {
        'dataset_drifted': dataset_drifted,
        'number_of_drifted_features': number_of_drifted_features,
        'share_of_drifted_features': share_of_drifted_features,
        'feature_drift_scores': feature_drift_scores,
        'full_report': {},
        'drift_table': {},
        'column_drifts': {},
        'method': 'statistical_fallback'
    }


def detect_drift_evidently(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
) -> Dict:
    """
    Detect data drift using Evidently AI.
    
    Args:
        reference_data: Reference (training) dataset
        current_data: Current (production) dataset
        threshold: Drift detection threshold
        
    Returns:
        Dictionary containing drift metrics and results
    """
    # If Evidently is not available, use statistical fallback
    if not HAS_EVIDENTLY or not HAS_EVIDENTLY_REPORT or Report is None:
        return _detect_drift_statistical(reference_data, current_data, threshold)
    
    try:
        if not HAS_DATA_DRIFT_TABLE or DataDriftTable is None:
            # Fallback to statistical tests
            return _detect_drift_statistical(reference_data, current_data, threshold)
        
        # Data drift table (primary method - works in all versions)
        data_drift_table = DataDriftTable()
        drift_table_report = Report(metrics=[data_drift_table])
        drift_table_report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        drift_table_result = drift_table_report.as_dict()
        
        # Extract feature-level drift scores from drift table
        feature_drift_scores = {}
        dataset_drifted = False
        number_of_drifted_features = 0
        
        if 'metrics' in drift_table_result and len(drift_table_result['metrics']) > 0:
            drift_table_data = drift_table_result['metrics'][0]['result'].get('drift_by_columns', {})
            for col, drift_info in drift_table_data.items():
                if isinstance(drift_info, dict):
                    drift_score = drift_info.get('drift_score', 0.0)
                    drift_detected = drift_info.get('drift_detected', False)
                    stat_test = drift_info.get('stattest_name', 'Unknown')
                    feature_drift_scores[col] = {
                        'drift_score': drift_score,
                        'drift_detected': drift_detected,
                        'stat_test': stat_test
                    }
                    if drift_detected:
                        number_of_drifted_features += 1
                        dataset_drifted = True
        
        # Calculate share of drifted features
        total_features = len(feature_drift_scores) if feature_drift_scores else 1
        share_of_drifted_features = number_of_drifted_features / total_features if total_features > 0 else 0.0
        
        # Try to get dataset-level drift metric if available
        dataset_drift_result = None
        if HAS_DATASET_DRIFT_METRIC:
            try:
                dataset_drift_metric = DatasetDriftMetric(threshold=threshold)
                dataset_drift_report = Report(metrics=[dataset_drift_metric])
                dataset_drift_report.run(
                    reference_data=reference_data,
                    current_data=current_data
                )
                dataset_drift_result = dataset_drift_report.as_dict()
                
                # Override with dataset-level metric if available
                if dataset_drift_result and 'metrics' in dataset_drift_result and len(dataset_drift_result['metrics']) > 0:
                    result_data = dataset_drift_result['metrics'][0]['result']
                    dataset_drifted = result_data.get('dataset_drift', dataset_drifted)
                    number_of_drifted_features = result_data.get('number_of_drifted_features', number_of_drifted_features)
                    share_of_drifted_features = result_data.get('share_of_drifted_features', share_of_drifted_features)
            except Exception:
                # If DatasetDriftMetric fails, use drift table results
                pass
        
        # Column-level drift metrics (optional, can be slow for many columns)
        column_drifts = {}
        if HAS_DATA_DRIFT_TABLE and ColumnDriftMetric is not None and Report is not None:
            try:
                # Only process numeric columns to avoid errors
                numeric_cols = reference_data.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 for performance
                for col in numeric_cols:
                    try:
                        col_drift_metric = ColumnDriftMetric(column_name=col)
                        col_report = Report(metrics=[col_drift_metric])
                        col_report.run(
                            reference_data=reference_data,
                            current_data=current_data
                        )
                        col_result = col_report.as_dict()
                        column_drifts[col] = col_result
                    except Exception:
                        continue
            except Exception:
                pass
        
        return {
            'dataset_drifted': dataset_drifted,
            'number_of_drifted_features': number_of_drifted_features,
            'share_of_drifted_features': share_of_drifted_features,
            'feature_drift_scores': feature_drift_scores,
            'full_report': dataset_drift_result if dataset_drift_result else drift_table_result,
            'drift_table': drift_table_result,
            'column_drifts': column_drifts,
            'method': 'evidently'
        }
    except Exception as e:
        # If Evidently fails, fall back to statistical tests
        try:
            return _detect_drift_statistical(reference_data, current_data, threshold)
        except Exception as e2:
            return {
                'error': f"Evidently error: {str(e)}, Statistical fallback error: {str(e2)}",
                'dataset_drifted': False,
                'number_of_drifted_features': 0,
                'share_of_drifted_features': 0.0,
                'feature_drift_scores': {},
                'method': 'error'
            }


def detect_drift_deepchecks(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Dict:
    """
    Detect data drift using Deepchecks (for validation).
    
    Args:
        reference_data: Reference (training) dataset
        current_data: Current (production) dataset
        target_column: Optional target column name
        
    Returns:
        Dictionary containing drift metrics and results
    """
    try:
        # Create Deepchecks Dataset objects
        cat_features = reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        ref_dataset = Dataset(
            reference_data,
            cat_features=cat_features if cat_features else None,
            label=target_column if target_column and target_column in reference_data.columns else None
        )
        
        curr_dataset = Dataset(
            current_data,
            cat_features=cat_features if cat_features else None,
            label=target_column if target_column and target_column in current_data.columns else None
        )
        
        # Run data drift check
        drift_check = DataDrift()
        drift_result = drift_check.run(ref_dataset, curr_dataset)
        result_dict = drift_result.to_json()
        result_dict_parsed = json.loads(result_dict)
        
        # Extract drift information
        drift_score = result_dict_parsed.get('value', {}).get('Drift score', 0.0)
        drift_detected = result_dict_parsed.get('value', {}).get('Drift detected', False)
        
        # Extract per-feature drift scores
        feature_drift_scores = {}
        if 'value' in result_dict_parsed and 'Drift score per feature' in result_dict_parsed['value']:
            per_feature = result_dict_parsed['value']['Drift score per feature']
            if isinstance(per_feature, dict):
                feature_drift_scores = per_feature
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'feature_drift_scores': feature_drift_scores,
            'full_result': result_dict_parsed
        }
    except Exception as e:
        return {
            'error': str(e),
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drift_scores': {}
        }


def calculate_model_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str,
    model=None,
) -> Dict:
    """
    Calculate model drift by training on reference and evaluating on current data.
    
    Args:
        reference_data: Reference (training) dataset
        current_data: Current (production) dataset
        target_column: Name of the target column
        model: Pre-trained model (if None, will train RandomForest)
        
    Returns:
        Dictionary containing model performance metrics
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        X_ref = reference_data.drop(columns=[target_column])
        y_ref = reference_data[target_column]
        X_curr = current_data.drop(columns=[target_column])
        y_curr = current_data[target_column]
        
        # Handle categorical features
        le = LabelEncoder()
        if y_ref.dtype == 'object' or y_ref.dtype.name == 'category':
            y_ref_encoded = le.fit_transform(y_ref)
            y_curr_encoded = le.transform(y_curr)
            is_classification = True
        else:
            y_ref_encoded = y_ref
            y_curr_encoded = y_curr
            is_classification = False
        
        # Encode categorical features in X
        for col in X_ref.columns:
            if X_ref[col].dtype == 'object' or X_ref[col].dtype.name == 'category':
                le_col = LabelEncoder()
                X_ref[col] = le_col.fit_transform(X_ref[col].astype(str))
                X_curr[col] = le_col.transform(X_curr[col].astype(str))
        
        # Train model if not provided
        if model is None:
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_ref, y_ref_encoded)
        
        # Evaluate on reference (training) data
        y_ref_pred = model.predict(X_ref)
        ref_accuracy = accuracy_score(y_ref_encoded, y_ref_pred) if is_classification else None
        ref_mse = mean_squared_error(y_ref_encoded, y_ref_pred) if not is_classification else None
        ref_mae = mean_absolute_error(y_ref_encoded, y_ref_pred) if not is_classification else None
        ref_r2 = r2_score(y_ref_encoded, y_ref_pred) if not is_classification else None
        
        # Classification metrics
        ref_precision = None
        ref_recall = None
        ref_f1 = None
        if is_classification:
            try:
                ref_precision = precision_score(y_ref_encoded, y_ref_pred, average='weighted', zero_division=0)
                ref_recall = recall_score(y_ref_encoded, y_ref_pred, average='weighted', zero_division=0)
                ref_f1 = f1_score(y_ref_encoded, y_ref_pred, average='weighted', zero_division=0)
            except:
                pass
        
        # Evaluate on current (production) data
        y_curr_pred = model.predict(X_curr)
        curr_accuracy = accuracy_score(y_curr_encoded, y_curr_pred) if is_classification else None
        curr_mse = mean_squared_error(y_curr_encoded, y_curr_pred) if not is_classification else None
        curr_mae = mean_absolute_error(y_curr_encoded, y_curr_pred) if not is_classification else None
        curr_r2 = r2_score(y_curr_encoded, y_curr_pred) if not is_classification else None
        
        # Classification metrics for current
        curr_precision = None
        curr_recall = None
        curr_f1 = None
        if is_classification:
            try:
                curr_precision = precision_score(y_curr_encoded, y_curr_pred, average='weighted', zero_division=0)
                curr_recall = recall_score(y_curr_encoded, y_curr_pred, average='weighted', zero_division=0)
                curr_f1 = f1_score(y_curr_encoded, y_curr_pred, average='weighted', zero_division=0)
            except:
                pass
        
        # Calculate performance drop
        if is_classification:
            performance_drop = ref_accuracy - curr_accuracy if ref_accuracy else 0.0
        else:
            performance_drop = curr_mse - ref_mse if ref_mse else 0.0
        
        return {
            'is_classification': is_classification,
            'reference_metrics': {
                'accuracy': ref_accuracy,
                'precision': ref_precision,
                'recall': ref_recall,
                'f1': ref_f1,
                'mse': ref_mse,
                'mae': ref_mae,
                'r2': ref_r2
            },
            'current_metrics': {
                'accuracy': curr_accuracy,
                'precision': curr_precision,
                'recall': curr_recall,
                'f1': curr_f1,
                'mse': curr_mse,
                'mae': curr_mae,
                'r2': curr_r2
            },
            'performance_drop': performance_drop,
            'model': model
        }
    except Exception as e:
        return {
            'error': str(e),
            'performance_drop': 0.0
        }


def get_top_drifting_features(
    feature_drift_scores: Dict,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Get top N drifting features sorted by drift score.
    
    Args:
        feature_drift_scores: Dictionary of feature drift scores
        top_n: Number of top features to return
        
    Returns:
        List of tuples (feature_name, drift_score) sorted by drift score
    """
    if not feature_drift_scores:
        return []
    
    # Handle different formats from Evidently and Deepchecks
    scores = []
    for feature, drift_info in feature_drift_scores.items():
        if isinstance(drift_info, dict):
            score = drift_info.get('drift_score', 0.0)
        elif isinstance(drift_info, (int, float)):
            score = float(drift_info)
        else:
            score = 0.0
        scores.append((feature, score))
    
    # Sort by drift score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def compare_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    method: str = 'evidently',
    threshold: float = 0.05
) -> Dict:
    """
    Compare drift between reference and current datasets.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        method: 'evidently' or 'deepchecks'
        threshold: Drift detection threshold
        
    Returns:
        Dictionary with drift comparison results
    """
    if method.lower() == 'evidently':
        return detect_drift_evidently(reference_data, current_data, threshold)
    elif method.lower() == 'deepchecks':
        target_col = None
        if 'target' in current_data.columns:
            target_col = 'target'
        return detect_drift_deepchecks(reference_data, current_data, target_col)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'evidently' or 'deepchecks'")


def compare_multiple_datasets(
    datasets: Dict[str, pd.DataFrame],
    reference_name: str,
    method: str = 'evidently',
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compare multiple datasets against a reference dataset.
    
    Args:
        datasets: Dictionary of {dataset_name: DataFrame}
        reference_name: Name of the reference dataset
        method: 'evidently' or 'deepchecks'
        threshold: Drift detection threshold
        
    Returns:
        DataFrame with drift scores per feature per dataset
    """
    if reference_name not in datasets:
        raise ValueError(f"Reference dataset '{reference_name}' not found in datasets")
    
    reference_data = datasets[reference_name]
    results = {}
    
    for name, current_data in datasets.items():
        if name == reference_name:
            continue
        
        try:
            drift_result = compare_drift(reference_data, current_data, method, threshold)
            feature_scores = drift_result.get('feature_drift_scores', {})
            
            # Extract scores
            scores_dict = {}
            for feature, drift_info in feature_scores.items():
                if isinstance(drift_info, dict):
                    scores_dict[feature] = drift_info.get('drift_score', 0.0)
                elif isinstance(drift_info, (int, float)):
                    scores_dict[feature] = float(drift_info)
                else:
                    scores_dict[feature] = 0.0
            
            results[name] = scores_dict
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        df.index.name = 'Feature'
        return df.fillna(0.0)
    else:
        return pd.DataFrame()


def generate_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    model=None,
    threshold: float = 0.05
) -> Dict:
    """
    Generate comprehensive drift report.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        model: Optional pre-trained model
        threshold: Drift detection threshold
        
    Returns:
        Dictionary with complete report data
    """
    report = {
        'data_drift': {},
        'model_drift': {},
        'summary': {}
    }
    
    # Data drift analysis
    try:
        data_drift = detect_drift_evidently(reference_data, current_data, threshold)
        report['data_drift'] = {
            'dataset_drifted': data_drift.get('dataset_drifted', False),
            'number_of_drifted_features': data_drift.get('number_of_drifted_features', 0),
            'share_of_drifted_features': data_drift.get('share_of_drifted_features', 0.0),
            'feature_drift_scores': data_drift.get('feature_drift_scores', {})
        }
    except Exception as e:
        report['data_drift'] = {'error': str(e)}
    
    # Model drift analysis (if target column exists)
    if 'target' in reference_data.columns and 'target' in current_data.columns:
        try:
            model_drift = calculate_model_drift(
                reference_data, current_data, 'target', model
            )
            report['model_drift'] = model_drift
        except Exception as e:
            report['model_drift'] = {'error': str(e)}
    
    # Summary
    report['summary'] = {
        'reference_shape': reference_data.shape,
        'current_shape': current_data.shape,
        'common_features': list(set(reference_data.columns) & set(current_data.columns))
    }
    
    return report


def calculate_feature_importance_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str,
    models: Optional[List] = None
) -> Dict:
    """
    Calculate feature importance drift across multiple models.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        target_column: Target column name
        models: List of model names or None for default models
        
    Returns:
        Dictionary with feature importance comparisons
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        
        if models is None:
            models = ['RandomForest', 'XGBoost', 'LogisticRegression']
        
        # Prepare data
        X_ref = reference_data.drop(columns=[target_column])
        y_ref = reference_data[target_column]
        X_curr = current_data.drop(columns=[target_column])
        y_curr = current_data[target_column]
        
        # Handle target encoding
        le = LabelEncoder()
        if y_ref.dtype == 'object' or y_ref.dtype.name == 'category':
            y_ref_encoded = le.fit_transform(y_ref)
            y_curr_encoded = le.transform(y_curr)
            is_classification = True
        else:
            y_ref_encoded = y_ref
            y_curr_encoded = y_curr
            is_classification = False
        
        # Encode categorical features
        for col in X_ref.columns:
            if X_ref[col].dtype == 'object' or X_ref[col].dtype.name == 'category':
                le_col = LabelEncoder()
                X_ref[col] = le_col.fit_transform(X_ref[col].astype(str))
                X_curr[col] = le_col.transform(X_curr[col].astype(str))
        
        importance_results = {}
        
        # Train models and get feature importance
        for model_name in models:
            try:
                if model_name == 'RandomForest':
                    if is_classification:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_ref, y_ref_encoded)
                    ref_importance = model.feature_importances_
                    
                    # Train on current for comparison
                    model_curr = type(model)(n_estimators=100, random_state=42)
                    model_curr.fit(X_curr, y_curr_encoded)
                    curr_importance = model_curr.feature_importances_
                
                elif model_name == 'XGBoost':
                    try:
                        if is_classification:
                            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                        else:
                            model = xgb.XGBRegressor(random_state=42)
                        model.fit(X_ref, y_ref_encoded)
                        ref_importance = model.feature_importances_
                        
                        model_curr = type(model)(random_state=42)
                        model_curr.fit(X_curr, y_curr_encoded)
                        curr_importance = model_curr.feature_importances_
                    except ImportError:
                        continue
                
                elif model_name == 'LogisticRegression':
                    if is_classification:
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(X_ref, y_ref_encoded)
                        ref_importance = np.abs(model.coef_[0])
                        
                        model_curr = LogisticRegression(random_state=42, max_iter=1000)
                        model_curr.fit(X_curr, y_curr_encoded)
                        curr_importance = np.abs(model_curr.coef_[0])
                    else:
                        continue
                else:
                    continue
                
                # Calculate importance drift
                importance_diff = np.abs(ref_importance - curr_importance)
                importance_results[model_name] = {
                    'reference_importance': dict(zip(X_ref.columns, ref_importance.tolist())),
                    'current_importance': dict(zip(X_curr.columns, curr_importance.tolist())),
                    'importance_drift': dict(zip(X_ref.columns, importance_diff.tolist()))
                }
            except Exception as e:
                importance_results[model_name] = {'error': str(e)}
        
        return importance_results
    except Exception as e:
        return {'error': str(e)}


def batch_model_comparison(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str,
    models: Optional[List[str]] = None
) -> Dict:
    """
    Compare multiple models' performance decay.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        target_column: Target column name
        models: List of model names or None for default
        
    Returns:
        Dictionary with model performance comparisons
    """
    if models is None:
        models = ['RandomForest', 'XGBoost', 'LogisticRegression']
    
    results = {}
    
    for model_name in models:
        try:
            if model_name == 'RandomForest':
                model = None  # Will be created in calculate_model_drift
            elif model_name == 'XGBoost':
                try:
                    import xgboost as xgb
                    # Will be handled in calculate_model_drift
                    model = None
                except ImportError:
                    results[model_name] = {'error': 'XGBoost not installed'}
                    continue
            elif model_name == 'LogisticRegression':
                model = None
            else:
                continue
            
            model_result = calculate_model_drift(
                reference_data, current_data, target_column, model
            )
            results[model_name] = model_result
        except Exception as e:
            results[model_name] = {'error': str(e)}
    
    return results


def export_drifted_features_csv(
    feature_drift_scores: Dict,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Export drifted features to CSV format.
    
    Args:
        feature_drift_scores: Dictionary of feature drift scores
        threshold: Threshold for drift detection
        
    Returns:
        DataFrame with drifted features
    """
    drifted_features = []
    
    for feature, drift_info in feature_drift_scores.items():
        if isinstance(drift_info, dict):
            drift_score = drift_info.get('drift_score', 0.0)
            drift_detected = drift_info.get('drift_detected', False)
            stat_test = drift_info.get('stat_test', 'Unknown')
        elif isinstance(drift_info, (int, float)):
            drift_score = float(drift_info)
            drift_detected = drift_score > threshold
            stat_test = 'Unknown'
        else:
            continue
        
        if drift_detected or drift_score > threshold:
            drifted_features.append({
                'Feature': feature,
                'Drift_Score': drift_score,
                'Drift_Detected': drift_detected,
                'Stat_Test': stat_test,
                'Threshold': threshold
            })
    
    if drifted_features:
        return pd.DataFrame(drifted_features).sort_values('Drift_Score', ascending=False)
    else:
        return pd.DataFrame(columns=['Feature', 'Drift_Score', 'Drift_Detected', 'Stat_Test', 'Threshold'])

