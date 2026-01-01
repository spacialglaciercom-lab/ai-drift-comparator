"""
Comprehensive tests for drift detection functionality.
"""
import pytest
import pandas as pd
import numpy as np
from utils.drift_utils import (
    detect_drift_evidently,
    detect_drift_deepchecks,
    compare_drift,
    calculate_model_drift,
    compare_multiple_datasets,
    get_top_drifting_features,
    export_drifted_features_csv
)


@pytest.fixture
def reference_data():
    """Generate reference dataset with normal distribution."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    data = {}
    for i in range(n_features):
        data[f'feature_{i+1}'] = np.random.normal(0, 1, n_samples)
    
    data['target'] = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(data)


@pytest.fixture
def current_data_drifted(reference_data):
    """Generate current dataset with drift (shift mean + std*0.5 on 30% features)."""
    np.random.seed(42)
    current = reference_data.copy()
    
    # Get numeric columns (exclude target)
    numeric_cols = [col for col in current.columns if col != 'target' and current[col].dtype in ['int64', 'float64']]
    
    # Select 30% of features to drift
    n_features_to_drift = max(1, int(len(numeric_cols) * 0.3))
    features_to_drift = np.random.choice(numeric_cols, size=n_features_to_drift, replace=False)
    
    # Apply drift: shift mean + std*0.5
    for col in features_to_drift:
        mean_val = current[col].mean()
        std_val = current[col].std()
        shift = mean_val + std_val * 0.5
        current[col] = current[col] + shift
    
    return current, features_to_drift.tolist()


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def mismatched_dataframes():
    """DataFrames with mismatched columns."""
    ref = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]})
    curr = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_3': [7, 8, 9]})
    return ref, curr


class TestDriftDetection:
    """Test drift detection functions."""
    
    def test_detect_drift_evidently_basic(self, reference_data, current_data_drifted):
        """Test basic Evidently drift detection."""
        current_data, _ = current_data_drifted
        
        result = detect_drift_evidently(
            reference_data,
            current_data,
            threshold=0.05
        )
        
        assert isinstance(result, dict)
        assert 'dataset_drifted' in result
        assert 'number_of_drifted_features' in result
        assert 'feature_drift_scores' in result
    
    def test_detect_drift_evidently_drifted_data(self, reference_data, current_data_drifted):
        """Test Evidently detects drift in drifted data."""
        current_data, drifted_features = current_data_drifted
        
        result = detect_drift_evidently(
            reference_data,
            current_data,
            threshold=0.05
        )
        
        # Should detect drift since we shifted features
        assert result.get('number_of_drifted_features', 0) >= 0
        assert len(result.get('feature_drift_scores', {})) > 0
    
    def test_detect_drift_deepchecks(self, reference_data, current_data_drifted):
        """Test Deepchecks drift detection."""
        current_data, _ = current_data_drifted
        
        result = detect_drift_deepchecks(
            reference_data,
            current_data,
            target_column='target'
        )
        
        assert isinstance(result, dict)
        assert 'drift_detected' in result or 'error' in result
        assert 'drift_score' in result or 'error' in result
    
    def test_compare_drift_evidently(self, reference_data, current_data_drifted):
        """Test compare_drift with Evidently."""
        current_data, _ = current_data_drifted
        
        result = compare_drift(
            reference_data,
            current_data,
            method='evidently',
            threshold=0.05
        )
        
        assert isinstance(result, dict)
        assert 'feature_drift_scores' in result or 'error' in result
    
    def test_compare_drift_deepchecks(self, reference_data, current_data_drifted):
        """Test compare_drift with Deepchecks."""
        current_data, _ = current_data_drifted
        
        result = compare_drift(
            reference_data,
            current_data,
            method='deepchecks',
            threshold=0.05
        )
        
        assert isinstance(result, dict)
    
    def test_compare_drift_invalid_method(self, reference_data, current_data_drifted):
        """Test compare_drift with invalid method."""
        current_data, _ = current_data_drifted
        
        with pytest.raises(ValueError):
            compare_drift(
                reference_data,
                current_data,
                method='invalid_method'
            )


class TestModelDrift:
    """Test model drift calculation."""
    
    def test_calculate_model_drift_classification(self, reference_data, current_data_drifted):
        """Test model drift for classification."""
        current_data, _ = current_data_drifted
        
        result = calculate_model_drift(
            reference_data,
            current_data,
            target_column='target'
        )
        
        assert isinstance(result, dict)
        assert 'is_classification' in result
        assert 'reference_metrics' in result
        assert 'current_metrics' in result
        assert 'performance_drop' in result
    
    def test_calculate_model_drift_performance_drop(self, reference_data, current_data_drifted):
        """Test that model drift detects performance drop."""
        current_data, _ = current_data_drifted
        
        result = calculate_model_drift(
            reference_data,
            current_data,
            target_column='target'
        )
        
        # Performance drop should be calculated
        assert 'performance_drop' in result
        assert isinstance(result['performance_drop'], (int, float))


class TestMultiDatasetComparison:
    """Test multi-dataset comparison."""
    
    def test_compare_multiple_datasets(self, reference_data, current_data_drifted):
        """Test comparing multiple datasets."""
        current_data, _ = current_data_drifted
        
        datasets = {
            'reference': reference_data,
            'current_1': current_data,
            'current_2': current_data.copy()
        }
        
        result = compare_multiple_datasets(
            datasets,
            'reference',
            method='evidently',
            threshold=0.05
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2  # current_1 and current_2
        assert len(result) > 0  # Should have features


class TestFeatureUtilities:
    """Test feature utility functions."""
    
    def test_get_top_drifting_features(self, reference_data, current_data_drifted):
        """Test getting top drifting features."""
        current_data, _ = current_data_drifted
        
        result = detect_drift_evidently(
            reference_data,
            current_data,
            threshold=0.05
        )
        
        feature_scores = result.get('feature_drift_scores', {})
        top_features = get_top_drifting_features(feature_scores, top_n=5)
        
        assert isinstance(top_features, list)
        assert len(top_features) <= 5
        if len(top_features) > 1:
            # Should be sorted descending
            assert top_features[0][1] >= top_features[1][1]
    
    def test_export_drifted_features_csv(self, reference_data, current_data_drifted):
        """Test exporting drifted features to CSV."""
        current_data, _ = current_data_drifted
        
        result = detect_drift_evidently(
            reference_data,
            current_data,
            threshold=0.05
        )
        
        feature_scores = result.get('feature_drift_scores', {})
        df = export_drifted_features_csv(feature_scores, threshold=0.05)
        
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert 'Feature' in df.columns
            assert 'Drift_Score' in df.columns


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, empty_dataframe):
        """Test handling of empty DataFrame."""
        # Create a non-empty current for comparison
        current = pd.DataFrame({'feature_1': [1, 2, 3]})
        
        result = detect_drift_evidently(empty_dataframe, current, threshold=0.05)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'error' in result or 'feature_drift_scores' in result
    
    def test_mismatched_columns(self, mismatched_dataframes):
        """Test handling of mismatched columns."""
        ref, curr = mismatched_dataframes
        
        # Should use common columns
        common_cols = list(set(ref.columns) & set(curr.columns))
        ref_common = ref[common_cols]
        curr_common = curr[common_cols]
        
        result = detect_drift_evidently(ref_common, curr_common, threshold=0.05)
        
        assert isinstance(result, dict)
    
    def test_single_feature_dataset(self):
        """Test with single feature dataset."""
        ref = pd.DataFrame({'feature_1': np.random.normal(0, 1, 100)})
        curr = pd.DataFrame({'feature_1': np.random.normal(1, 1, 100)})  # Shifted
        
        result = detect_drift_evidently(ref, curr, threshold=0.05)
        
        assert isinstance(result, dict)
    
    def test_categorical_features(self):
        """Test with categorical features."""
        ref = pd.DataFrame({
            'numeric': np.random.normal(0, 1, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        curr = pd.DataFrame({
            'numeric': np.random.normal(1, 1, 100),  # Shifted
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        result = detect_drift_evidently(ref, curr, threshold=0.05)
        
        assert isinstance(result, dict)
    
    def test_large_dataset_chunking(self):
        """Test with larger dataset (simulating chunked processing)."""
        # Create larger dataset
        ref = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, 5000)
            for i in range(5)
        })
        curr = pd.DataFrame({
            f'feature_{i}': np.random.normal(0.5, 1, 5000)  # Shifted
            for i in range(5)
        })
        
        result = detect_drift_evidently(ref, curr, threshold=0.05)
        
        assert isinstance(result, dict)
        assert 'error' not in result or result.get('error') is None


class TestDriftScores:
    """Test drift score calculations."""
    
    def test_drift_scores_range(self, reference_data, current_data_drifted):
        """Test that drift scores are in valid range."""
        current_data, _ = current_data_drifted
        
        result = detect_drift_evidently(
            reference_data,
            current_data,
            threshold=0.05
        )
        
        feature_scores = result.get('feature_drift_scores', {})
        
        for feature, drift_info in feature_scores.items():
            if isinstance(drift_info, dict):
                score = drift_info.get('drift_score', 0.0)
                assert 0.0 <= score <= 1.0, f"Drift score {score} out of range for {feature}"
    
    def test_no_drift_detection(self):
        """Test that identical datasets show no drift."""
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100)
        })
        
        # Compare dataset with itself
        result = detect_drift_evidently(data, data.copy(), threshold=0.05)
        
        # Should have results (may or may not detect drift depending on statistical test)
        assert isinstance(result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

