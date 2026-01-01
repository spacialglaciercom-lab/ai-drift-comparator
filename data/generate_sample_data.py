"""
Generate sample datasets for drift detection testing.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, fetch_california_housing
from sklearn.preprocessing import StandardScaler


def generate_classification_data(n_samples=1000, n_features=10, random_state=42):
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def generate_drifted_data(reference_df: pd.DataFrame, drift_percentage=0.2, random_state=42):
    """
    Generate drifted version of reference data by shifting 20% of features.
    
    Args:
        reference_df: Reference DataFrame
        drift_percentage: Percentage of features to drift (default 0.2 = 20%)
        random_state: Random seed
        
    Returns:
        Drifted DataFrame
    """
    np.random.seed(random_state)
    drifted_df = reference_df.copy()
    
    # Get numeric columns (exclude target if present)
    numeric_cols = drifted_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    # Select features to drift
    n_features_to_drift = max(1, int(len(numeric_cols) * drift_percentage))
    features_to_drift = np.random.choice(numeric_cols, size=n_features_to_drift, replace=False)
    
    # Apply drift: shift mean and add noise
    for col in features_to_drift:
        mean_shift = np.random.uniform(-2, 2)
        std_multiplier = np.random.uniform(1.2, 2.0)
        
        # Shift the distribution
        drifted_df[col] = drifted_df[col] + mean_shift
        # Increase variance
        drifted_df[col] = drifted_df[col] * std_multiplier
        # Add noise
        noise = np.random.normal(0, 0.5, size=len(drifted_df))
        drifted_df[col] = drifted_df[col] + noise
    
    return drifted_df, features_to_drift.tolist()


def generate_demo_data(n_samples=1000, n_features=10, random_state=42):
    """
    Generate demo data for testing: reference (normal dist) and current (drifted).
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        Tuple of (reference_data, current_data, drifted_features)
    """
    np.random.seed(random_state)
    
    # Generate reference data with normal distribution
    reference_data = pd.DataFrame({
        f'feature_{i+1}': np.random.normal(0, 1, n_samples)
        for i in range(n_features)
    })
    reference_data['target'] = np.random.randint(0, 2, n_samples)
    
    # Generate current data with drift on 30% of features
    current_data = reference_data.copy()
    
    # Select 30% of features to drift
    numeric_cols = [col for col in reference_data.columns if col != 'target']
    n_features_to_drift = max(1, int(len(numeric_cols) * 0.3))
    features_to_drift = np.random.choice(numeric_cols, size=n_features_to_drift, replace=False)
    
    # Apply drift: shift mean + std*0.5
    for col in features_to_drift:
        mean_val = reference_data[col].mean()
        std_val = reference_data[col].std()
        shift = mean_val + std_val * 0.5
        current_data[col] = current_data[col] + shift
    
    return reference_data, current_data, features_to_drift.tolist()


def generate_timeseries_data(n_samples=1000, n_features=5, random_state=42, start_date='2023-01-01'):
    """
    Generate time-series dataset with potential drift.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        start_date: Start date for time series
        
    Returns:
        DataFrame with timestamp and features
    """
    np.random.seed(random_state)
    
    # Generate timestamps
    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    # Generate features with some trend and seasonality
    data = {'timestamp': dates}
    
    for i in range(n_features):
        # Base trend
        trend = np.linspace(0, 2, n_samples)
        # Seasonality
        seasonality = np.sin(2 * np.pi * np.arange(n_samples) / 365) * 0.5
        # Noise
        noise = np.random.normal(0, 0.3, n_samples)
        
        data[f'feature_{i+1}'] = trend + seasonality + noise
    
    # Add target (classification)
    data['target'] = (np.random.random(n_samples) > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    return df


def load_adult_income_dataset():
    """Load or generate adult income dataset."""
    try:
        # Try to load from sklearn (if available in future versions)
        # For now, generate a similar dataset
        from sklearn.datasets import fetch_openml
        adult = fetch_openml(name='adult', version=2, as_frame=True, parser='pandas')
        df = adult.frame
        
        # Clean and prepare
        df = df.dropna()
        # Encode target
        df['target'] = (df['income'] == '>50K').astype(int)
        df = df.drop(columns=['income'])
        
        return df
    except:
        # Fallback: generate synthetic adult-like dataset
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'workclass': np.random.choice(['Private', 'Self-emp', 'Government', 'Other'], n_samples),
            'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Doctorate', 'Some-college'], n_samples),
            'education-num': np.random.randint(9, 16, n_samples),
            'marital-status': np.random.choice(['Married', 'Divorced', 'Never-married', 'Widowed'], n_samples),
            'occupation': np.random.choice(['Tech', 'Sales', 'Service', 'Exec', 'Other'], n_samples),
            'relationship': np.random.choice(['Husband', 'Wife', 'Own-child', 'Other'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'capital-gain': np.random.exponential(100, n_samples),
            'capital-loss': np.random.exponential(50, n_samples),
            'hours-per-week': np.random.randint(20, 60, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target based on some features
        df['target'] = (
            (df['age'] > 35) & 
            (df['education-num'] > 12) & 
            (df['hours-per-week'] > 40)
        ).astype(int)
        
        return df


if __name__ == '__main__':
    # Generate sample datasets
    print("Generating sample datasets...")
    
    # Classification dataset
    ref_data = generate_classification_data(n_samples=1000, n_features=10)
    ref_data.to_csv('reference_classification.csv', index=False)
    print("Generated reference_classification.csv")
    
    # Drifted version
    curr_data, drifted_features = generate_drifted_data(ref_data, drift_percentage=0.2)
    curr_data.to_csv('current_classification.csv', index=False)
    print(f"Generated current_classification.csv (drifted features: {drifted_features})")
    
    # Adult income dataset
    adult_data = load_adult_income_dataset()
    adult_data.to_csv('reference_adult.csv', index=False)
    print("Generated reference_adult.csv")
    
    # Drifted adult data
    adult_drifted, adult_drifted_features = generate_drifted_data(adult_data, drift_percentage=0.2)
    adult_drifted.to_csv('current_adult.csv', index=False)
    print(f"Generated current_adult.csv (drifted features: {adult_drifted_features})")
    
    print("\nAll sample datasets generated successfully!")

