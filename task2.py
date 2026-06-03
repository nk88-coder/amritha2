# ==============================================================
# VS CODE INFERENCE — ALL 3 MODELS (with DAY feature)
# Put this file in the SAME folder as:
#   irradiance_model.pt  temperature_model.pt  windspeed_model.pt
#   scaler_irradiance.pkl  scaler_temperature.pkl  scaler_windspeed.pkl
#   scaler_t2m.pkl  state_encoder.pkl
# Then run: python inference.py
# ==============================================================

import math
import warnings
import torch
import torch.nn as nn
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ==============================================================
# MODEL DEFINITIONS — must match training exactly
# BASE_DIM=8, IRR_DIM=9
# ==============================================================

class WeatherModel(nn.Module):

    def __init__(self, num_states, input_dim, emb_dim=16, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(num_states, emb_dim)
        self.lstm = nn.LSTM(input_size=input_dim+emb_dim, hidden_size=hidden,
                            num_layers=2, dropout=0.2, batch_first=True)
        self.gru  = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),     nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),      nn.Sigmoid(),
        )

    def forward(self, features, states):
        emb  = self.embedding(states)
        x    = torch.cat([features, emb], dim=-1)
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        x    = x[:, -1]
        return self.head(x).squeeze(-1)


class IrradianceModel(nn.Module):

    def __init__(self, num_states, input_dim, emb_dim=32, hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(num_states, emb_dim)
        self.lstm = nn.LSTM(input_size=input_dim+emb_dim, hidden_size=hidden,
                            num_layers=3, dropout=0.2, batch_first=True)
        self.gru  = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),      nn.ReLU(),
            nn.Linear(32, 1),       nn.Sigmoid(),
        )

    def forward(self, features, states):
        emb  = self.embedding(states)
        x    = torch.cat([features, emb], dim=-1)
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        x    = x[:, -1]
        return self.head(x).squeeze(-1)

# ==============================================================
# LOAD ALL FILES
# ==============================================================

DEVICE = torch.device("cpu")

print("Loading files...")

state_encoder = joblib.load("state_encoder.pkl")
scaler_irr    = joblib.load("scaler_irradiance.pkl")
scaler_temp   = joblib.load("scaler_temperature.pkl")
scaler_wind   = joblib.load("scaler_windspeed.pkl")
t2m_scaler    = joblib.load("scaler_t2m.pkl")
NUM_STATES    = len(state_encoder.classes_)

irr_model = IrradianceModel(NUM_STATES, input_dim=9).to(DEVICE)
irr_model.load_state_dict(torch.load("irradiance_model.pt", map_location=DEVICE))
irr_model.eval()

temp_model = WeatherModel(NUM_STATES, input_dim=8).to(DEVICE)
temp_model.load_state_dict(torch.load("temperature_model.pt", map_location=DEVICE))
temp_model.eval()

wind_model = WeatherModel(NUM_STATES, input_dim=8).to(DEVICE)
wind_model.load_state_dict(torch.load("windspeed_model.pt", map_location=DEVICE))
wind_model.eval()

print(f"All models loaded. {NUM_STATES} states available.")
print(f"States: {list(state_encoder.classes_)}\n")

# ==============================================================
# FEATURE BUILDER — now includes day
# ==============================================================

def build_base_features(month, day, hour, rh2m):
    """8 base features for temp + windspeed models."""
    return [
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        math.sin(2 * math.pi * day   / 31),
        math.cos(2 * math.pi * day   / 31),
        math.sin(2 * math.pi * hour  / 24),
        math.cos(2 * math.pi * hour  / 24),
        rh2m,
        float(6 <= hour <= 18),
    ]

# ==============================================================
# PREDICT FUNCTIONS
# ==============================================================

def _run_model(model, features_list, state_id):
    feat  = torch.tensor(
        np.array([features_list], dtype=np.float32)
    ).unsqueeze(0).to(DEVICE)                       # [1, 1, N]
    state = torch.tensor([[state_id]]).to(DEVICE)   # [1, 1]
    with torch.no_grad():
        return model(feat, state).item()


def predict_irradiance(state: str, month: int, day: int, hour: int, rh2m: float, t2m: float):
    """Returns irradiance in W/m2."""
    state_id  = int(state_encoder.transform([state])[0])
    base      = build_base_features(month, day, hour, rh2m)
    t2m_scaled = float(t2m_scaler.transform(np.array([[t2m]]))[0][0])
    scaled    = _run_model(irr_model, base + [t2m_scaled], state_id)
    return round(float(scaler_irr.inverse_transform([[scaled]])[0][0]), 2)


def predict_temperature(state: str, month: int, day: int, hour: int, rh2m: float):
    """Returns temperature in degrees C."""
    state_id = int(state_encoder.transform([state])[0])
    base     = build_base_features(month, day, hour, rh2m)
    scaled   = _run_model(temp_model, base, state_id)
    return round(float(scaler_temp.inverse_transform([[scaled]])[0][0]), 2)


def predict_windspeed(state: str, month: int, day: int, hour: int, rh2m: float):
    """Returns wind speed in m/s."""
    state_id = int(state_encoder.transform([state])[0])
    base     = build_base_features(month, day, hour, rh2m)
    scaled   = _run_model(wind_model, base, state_id)
    return round(float(scaler_wind.inverse_transform([[scaled]])[0][0]), 2)


def predict_all(state: str, month: int, day: int, hour: int, rh2m: float, t2m: float):
    """Returns all 3 predictions at once."""
    return {
        "irradiance"  : predict_irradiance(state, month, day, hour, rh2m, t2m),
        "temperature" : predict_temperature(state, month, day, hour, rh2m),
        "windspeed"   : predict_windspeed(state, month, day, hour, rh2m),
    }

# ==============================================================
# RUN — edit these values
# ==============================================================

if __name__ == "__main__":

    STATE = "Karnataka"
    MONTH = 10
    DAY   = 15
    RH2M  = 55.0
    T2M   = 34.0

    # -- single prediction
    print("="*50)
    print(f"Single prediction — {STATE}, {DAY}/{MONTH}, 4 PM")
    print("="*50)
    result = predict_all(state=STATE, month=MONTH, day=DAY, hour=16, rh2m=RH2M, t2m=T2M)
    print(f"  Irradiance  : {result['irradiance']}  W/m2")
    print(f"  Temperature : {result['temperature']} C")
    print(f"  Wind speed  : {result['windspeed']}  m/s")

    # -- hourly sweep
    print(f"\nHourly sweep — {STATE}, {DAY}/{MONTH}")
    print("-"*50)
    for hr in range(6, 20):
        r   = predict_all(state=STATE, month=MONTH, day=DAY, hour=hr, rh2m=RH2M, t2m=T2M)
        bar = "█" * int(r['irradiance'] / 30)
        print(f"  {hr:02d}:00 | {r['irradiance']:6.1f} W/m2 | {r['temperature']}C | {r['windspeed']} m/s  {bar}")