import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from data import load_data

_ggcn_model  = None
_data        = None
_test_preds  = None
_test_actual = None
_ready       = False

# ═══════════════════════════════════════════════════════════════════
# GGCN Architecture  (exact copy from notebook)
# ═══════════════════════════════════════════════════════════════════

class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.2):
        super().__init__()
        self.conv      = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn        = nn.BatchNorm1d(out_channels)
        self.gate_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.dropout   = nn.Dropout(dropout)
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        features = F.relu(self.bn(self.conv(x)))
        gate     = torch.sigmoid(self.gate_conv(x))
        x_res    = self.residual_conv(x) if self.residual_conv else x
        return self.dropout((features * gate) + x_res)


class GGCNModel(nn.Module):
    def __init__(self, num_features,
                 conv1_filters=96, conv2_filters=128,
                 graph_filters=160, dense_units=128, dropout=0.2):
        super().__init__()
        self.num_features = num_features

        self.conv1 = nn.Conv1d(num_features, conv1_filters, 1, padding='same')
        self.bn1   = nn.BatchNorm1d(conv1_filters)
        self.conv2 = nn.Conv1d(conv1_filters, conv1_filters, 1, padding='same')
        self.bn2   = nn.BatchNorm1d(conv1_filters)
        self.dropout1 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(conv1_filters, conv2_filters, 1, padding='same')
        self.bn3   = nn.BatchNorm1d(conv2_filters)
        self.conv4 = nn.Conv1d(conv2_filters, conv2_filters, 1, padding='same')
        self.bn4   = nn.BatchNorm1d(conv2_filters)
        self.dropout2 = nn.Dropout(dropout)

        self.graph_conv1 = GatedConvBlock(conv2_filters, graph_filters,      1, dropout * 1.5)
        self.graph_conv2 = GatedConvBlock(graph_filters, graph_filters // 2, 1, dropout * 1.5)

        pool_out = graph_filters // 2
        self.fc1 = nn.Linear(pool_out,        dense_units)
        self.dropout3 = nn.Dropout(dropout * 1.5)
        self.fc2 = nn.Linear(dense_units,     dense_units // 2)
        self.dropout4 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(dense_units // 2, dense_units // 4)
        self.dropout5 = nn.Dropout(dropout * 0.5)
        self.fc4 = nn.Linear(dense_units // 4, 1)

    def forward(self, x):
        # x: (batch, seq_len=1, num_features)
        x = x.permute(0, 2, 1)                       # → (batch, features, seq)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(x)
        x = self.graph_conv1(x)
        x = self.graph_conv2(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # global avg pool
        x = F.relu(self.fc1(x));  x = self.dropout3(x)
        x = F.relu(self.fc2(x));  x = self.dropout4(x)
        x = F.relu(self.fc3(x));  x = self.dropout5(x)
        return self.fc4(x)


# ═══════════════════════════════════════════════════════════════════
# Load model + run inference on test set
# ═══════════════════════════════════════════════════════════════════

def load_model():
    global _ggcn_model, _data, _test_preds, _test_actual, _ready
    if _ready:
        return

    _data = load_data()
    num_features = 5  # Exact value from miaf_ggcn_model_info.json
    device = torch.device('cpu')

    print(f"🔄 Loading GGCN model ({num_features} features)...")
    model = GGCNModel(num_features=num_features).to(device)

    state = torch.load('data/ggcn_rul_model.pth', map_location=device, weights_only=True)
    # Handle various save formats
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    _ggcn_model = model

    # Run predictions on held-out test set
    X_test = _data['X_test']
    y_test = _data['y_test']

    X_tensor = torch.FloatTensor(X_test.reshape(-1, 1, num_features))
    with torch.no_grad():
        preds_norm = _ggcn_model(X_tensor).cpu().numpy().flatten()

    y_scaler = _data['y_scaler']
    _test_preds  = y_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
    _test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    r2   = metrics.r2_score(_test_actual, _test_preds)
    mape = metrics.mean_absolute_percentage_error(_test_actual, _test_preds) * 100
    print(f"  ✅ GGCN loaded — R²: {round(r2*100,2)}% | MAPE: {round(mape,2)}%")

    _ready = True
    print("✅ Model ready!")


# ═══════════════════════════════════════════════════════════════════
# API helpers
# ═══════════════════════════════════════════════════════════════════

def get_evaluation():
    load_model()
    from data import MODEL_INFO
    r2   = metrics.r2_score(_test_actual, _test_preds)
    mae  = metrics.mean_absolute_error(_test_actual, _test_preds)
    rmse = np.sqrt(metrics.mean_squared_error(_test_actual, _test_preds))
    mape = metrics.mean_absolute_percentage_error(_test_actual, _test_preds) * 100
    return {
        'r2':             round(r2 * 100, 2),
        'mae':            round(mae, 1),
        'rmse':           round(rmse, 1),
        'mape':           round(mape, 2),
        'num_features':   MODEL_INFO['num_features'],
        'feature_groups': MODEL_INFO['feature_groups'],
        'd_z':            MODEL_INFO['d_z'],
        'd_h':            MODEL_INFO['d_h'],
        'model':          'GGCN · Gated Graph Conv Network',
        'best_features':  _data['best_features'],
    }


def get_forecast(hours=48):
    load_model()
    n = min(hours, len(_test_preds))
    result = []
    for i in range(n):
        result.append({
            'index':     i,
            'predicted': round(float(_test_preds[i]), 1),
            'actual':    round(float(_test_actual[i]), 1),
        })
    return result


def get_realtime():
    load_model()
    df       = _data['df']
    y_scaler = _data['y_scaler']
    best_f   = _data['best_features']

    # Latest row
    last_row = df.tail(1)
    avg_load = float(df['Electricity Load'].mean())
    last_val = float(last_row['Electricity Load'].iloc[0])

    # Predict on last row
    X_scaler    = _data['X_scaler']
    num_features = len(best_f)

    X_raw  = last_row[best_f].values.astype(float)
    X_norm = X_scaler.transform(X_raw)
    X_t = torch.FloatTensor(X_norm.reshape(-1, 1, num_features))
    with torch.no_grad():
        pred_norm = _ggcn_model(X_t).cpu().numpy().flatten()
    pred_val = float(y_scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()[0])

    pct = round((last_val - avg_load) / avg_load * 100, 1)

    return {
        'actual':      round(last_val, 1),
        'predicted':   round(pred_val, 1),
        'avg':         round(avg_load, 1),
        'pct_vs_avg':  pct,
        'unit':        'MW',
        'timestamp':   str(df.index[-1]) if hasattr(df.index, '__len__') else 'N/A',
    }


def get_anomalies(limit=50):
    load_model()
    df       = _data['df']
    y_scaler = _data['y_scaler']
    best_f   = _data['best_features']
    X_scaler = _data['X_scaler']
    num_features = len(best_f)

    # Use last 500 rows for anomaly detection
    sample = df.tail(500).copy()
    X_raw  = sample[best_f].values.astype(float)
    X_norm = X_scaler.transform(X_raw)
    X_t    = torch.FloatTensor(X_norm.reshape(-1, 1, num_features))

    with torch.no_grad():
        preds_norm = _ggcn_model(X_t).cpu().numpy().flatten()
    preds = y_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
    actuals = sample['Electricity Load'].values

    errors = np.abs(preds - actuals)
    threshold = np.percentile(errors, 95)
    anomaly_mask = errors > threshold

    result = []
    timestamps = sample.index if hasattr(sample.index, '__iter__') else range(len(sample))
    for i, (ts, is_anom) in enumerate(zip(timestamps, anomaly_mask)):
        if is_anom:
            result.append({
                'datetime':  str(ts),
                'actual':    round(float(actuals[i]), 1),
                'predicted': round(float(preds[i]), 1),
                'error':     round(float(errors[i]), 1),
                'severity':  'CRITICAL' if errors[i] > threshold * 1.5 else 'HIGH',
            })

    return result[-limit:]