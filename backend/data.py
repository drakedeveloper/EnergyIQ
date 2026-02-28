import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

_df            = None
_X_scaler      = None
_y_scaler      = None
_best_features = None
_feature_cols  = None
_X_test        = None
_y_test        = None
_loaded        = False

# ── Exact features selected by MIAF (from miaf_ggcn_model_info.json) ─
BEST_FEATURES = [
    "Net Load",
    "Renewable Energy Load",
    "Is Weekend",
    "Historical Electricity Load (kW)",
    "Current Level (A)",
]

MODEL_INFO = {
    "num_features":       5,
    "test_rmse_original": 33.65,
    "test_mae_original":  23.41,
    "test_r2":            0.9835,
    "feature_groups":     21,
    "d_z":                252,
    "d_h":                80,
}

TARGET_COL = "Electricity Load"


def load_data():
    global _df, _X_scaler, _y_scaler, _best_features, _feature_cols
    global _X_test, _y_test, _loaded

    if _loaded:
        return {
            'df':            _df,
            'X_scaler':      _X_scaler,
            'y_scaler':      _y_scaler,
            'best_features': _best_features,
            'feature_cols':  _feature_cols,
            'X_test':        _X_test,
            'y_test':        _y_test,
            'model_info':    MODEL_INFO,
        }

    print("📦 Loading ISONE smart city dataset...")
    df = pd.read_csv('data/smart_city_energy_dataset.csv')

    # Parse timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')

    # All numeric feature columns (excluding target)
    X_df = df.drop(columns=[TARGET_COL], errors='ignore').select_dtypes(include=[np.number])
    feature_cols = X_df.columns.tolist()
    X = X_df.values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    print(f"  ✅ {X.shape[0]:,} rows · {X.shape[1]} total features")
    print(f"  📋 Using {len(BEST_FEATURES)} MIAF-selected features")

    # ── Fit scalers on full dataset (same as training) ────────────────
    # IMPORTANT: fit on selected features only, same order as training
    best_indices = [feature_cols.index(f) for f in BEST_FEATURES]
    X_selected   = X[:, best_indices]

    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    X_normalized = X_scaler.fit_transform(X_selected)
    y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Reproduce the same test split as training (test_size=0.15, random_state=42)
    _, X_test, _, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.15, random_state=42
    )

    _df            = df
    _X_scaler      = X_scaler
    _y_scaler      = y_scaler
    _best_features = BEST_FEATURES
    _feature_cols  = feature_cols
    _X_test        = X_test
    _y_test        = y_test
    _loaded        = True

    print(f"✅ Data ready · features: {BEST_FEATURES}")
    return {
        'df':            _df,
        'X_scaler':      _X_scaler,
        'y_scaler':      _y_scaler,
        'best_features': _best_features,
        'feature_cols':  _feature_cols,
        'X_test':        _X_test,
        'y_test':        _y_test,
        'model_info':    MODEL_INFO,
    }