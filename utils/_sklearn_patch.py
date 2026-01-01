"""
Patch for sklearn scorers and NumPy compatibility - must be imported before deepchecks or evidently.
This fixes the 'max_error' scorer issue and np.Inf compatibility.
"""
# CRITICAL: This must run BEFORE importing deepchecks or evidently
# These libraries try to use 'max_error' scorer at import time

# Fix for NumPy 2.0 compatibility: np.Inf was removed, use np.inf instead
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

try:
    from sklearn.metrics._scorer import _SCORERS
    from sklearn.metrics import make_scorer
    
    # Register 'max_error' as a scorer if it doesn't exist
    if 'max_error' not in _SCORERS:
        try:
            # Try to import max_error function directly
            from sklearn.metrics import max_error as max_error_func
            max_error_scorer = make_scorer(max_error_func, greater_is_better=False)
            _SCORERS['max_error'] = max_error_scorer
        except ImportError:
            # If max_error function doesn't exist, create a simple implementation
            def max_error_func(y_true, y_pred):
                """Calculate maximum error between true and predicted values."""
                return np.max(np.abs(y_true - y_pred))
            max_error_scorer = make_scorer(max_error_func, greater_is_better=False)
            _SCORERS['max_error'] = max_error_scorer
except Exception:
    # If sklearn.metrics is not available, continue without the fix
    pass

