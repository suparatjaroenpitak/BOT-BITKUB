"""
LSTM Price Prediction Model - พยากรณ์ราคา Crypto ด้วย LSTM
"""
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from config import AIConfig
from utils.logger import TradeLogger


class LSTMPriceModel(nn.Module):
    """LSTM Neural Network for price prediction."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class LSTMPredictor:
    """LSTM-based price predictor with training and inference."""

    def __init__(self, config: AIConfig, logger: Optional[TradeLogger] = None):
        self.config = config
        self.logger = logger or TradeLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMPriceModel] = None
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            "close", "volume", "rsi", "macd", "macd_signal",
            "bb_upper", "bb_lower", "ema_9", "ema_21",
        ]

    def _get_available_feature_columns(self, df: pd.DataFrame) -> list:
        """Return model features that exist in the provided dataframe."""
        return [column for column in self.feature_columns if column in df.columns]

    def _fit_scaler_from_df(self, df: Optional[pd.DataFrame]) -> bool:
        """Fit the scaler from runtime market data when persisted state is unavailable."""
        if df is None:
            return False

        available_cols = self._get_available_feature_columns(df)
        if not available_cols:
            return False

        data = df[available_cols].dropna().values
        if len(data) == 0:
            return False

        self.scaler.fit(data)
        return True

    def _is_scaler_fitted(self) -> bool:
        """Check whether the scaler has been fitted or restored."""
        return hasattr(self.scaler, "scale_") and hasattr(self.scaler, "min_")

    def _restore_scaler_from_checkpoint(self, checkpoint: dict) -> bool:
        """Restore persisted MinMaxScaler state without requiring live data."""
        scaler_min = checkpoint.get("scaler_min") or []
        scaler_max = checkpoint.get("scaler_max") or []
        scaler_scale = checkpoint.get("scaler_scale") or []
        scaler_offset = checkpoint.get("scaler_offset") or []
        scaler_range = checkpoint.get("scaler_range") or []

        if not (scaler_min and scaler_max and scaler_scale):
            return False

        data_min = np.asarray(scaler_min, dtype=np.float64)
        data_max = np.asarray(scaler_max, dtype=np.float64)
        scale = np.asarray(scaler_scale, dtype=np.float64)

        if not (len(data_min) == len(data_max) == len(scale)):
            return False

        data_range = np.asarray(scaler_range, dtype=np.float64) if scaler_range else (data_max - data_min)
        min_offset = np.asarray(scaler_offset, dtype=np.float64) if scaler_offset else (-data_min * scale)

        self.scaler.data_min_ = data_min
        self.scaler.data_max_ = data_max
        self.scaler.data_range_ = data_range
        self.scaler.scale_ = scale
        self.scaler.min_ = min_offset
        self.scaler.n_features_in_ = len(data_min)
        self.scaler.n_samples_seen_ = checkpoint.get("scaler_samples_seen", 0)
        return True

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        # Select features
        available_cols = self._get_available_feature_columns(df)
        data = df[available_cols].dropna().values

        # Normalize
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        seq_len = self.config.lstm_sequence_length

        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i - seq_len:i])
            # Predict the close price (index 0)
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> dict:
        """Train the LSTM model."""
        X, y = self.prepare_data(df)
        if len(X) == 0:
            self.logger.log_error("LSTM training", ValueError("Not enough data"))
            return {"error": "Not enough data for training"}

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.lstm_batch_size, shuffle=True
        )

        # Initialize model
        input_size = X_train.shape[2]
        self.model = LSTMPriceModel(
            input_size=input_size,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lstm_learning_rate
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        self.logger.log_info(f"Starting LSTM training: {self.config.lstm_epochs} epochs")

        for epoch in range(self.config.lstm_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()

            if (epoch + 1) % 10 == 0:
                self.logger.log_info(
                    f"Epoch {epoch + 1}/{self.config.lstm_epochs} | "
                    f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}"
                )

        # Keep the in-memory predictor aligned with the persisted best checkpoint.
        if os.path.exists(self.config.lstm_model_path):
            self.load_model(df)

        self.logger.log_info(f"LSTM training complete. Best Val Loss: {best_val_loss:.6f}")
        return history

    def predict(self, df: pd.DataFrame) -> dict:
        """Predict next price movement."""
        if self.model is None:
            self.load_model(df)
            if self.model is None:
                return {"predicted_price": 0, "direction": "unknown", "confidence": 0}

        available_cols = self._get_available_feature_columns(df)
        data = df[available_cols].dropna().values

        if len(data) < self.config.lstm_sequence_length:
            return {"predicted_price": 0, "direction": "unknown", "confidence": 0}

        if not self._is_scaler_fitted() and not self._fit_scaler_from_df(df):
            self.logger.log_error("LSTM predict", ValueError("Scaler is not fitted"))
            return {"predicted_price": 0, "direction": "unknown", "confidence": 0}

        # Normalize using fitted scaler
        scaled_data = self.scaler.transform(data)

        # Take last sequence
        seq = scaled_data[-self.config.lstm_sequence_length:]
        X = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X).item()

        # Inverse transform prediction
        dummy = np.zeros((1, len(available_cols)))
        dummy[0, 0] = prediction
        predicted_price = self.scaler.inverse_transform(dummy)[0, 0]

        current_price = df["close"].iloc[-1]
        price_change_pct = ((predicted_price - current_price) / current_price) * 100

        direction = "up" if predicted_price > current_price else "down"
        confidence = min(abs(price_change_pct) / 5.0, 1.0)  # Normalize confidence

        result = {
            "predicted_price": predicted_price,
            "current_price": current_price,
            "price_change_pct": price_change_pct,
            "direction": direction,
            "confidence": confidence,
        }

        self.logger.log_ai_prediction(
            symbol="", current_price=current_price,
            predicted_price=predicted_price, confidence=confidence,
            direction=direction
        )

        return result

    def save_model(self):
        """Save model to disk."""
        if self.model is None:
            return
        os.makedirs(os.path.dirname(self.config.lstm_model_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "scaler_min": self.scaler.data_min_.tolist() if hasattr(self.scaler, "data_min_") else [],
            "scaler_max": self.scaler.data_max_.tolist() if hasattr(self.scaler, "data_max_") else [],
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else [],
            "scaler_offset": self.scaler.min_.tolist() if hasattr(self.scaler, "min_") else [],
            "scaler_range": self.scaler.data_range_.tolist() if hasattr(self.scaler, "data_range_") else [],
            "scaler_samples_seen": getattr(self.scaler, "n_samples_seen_", 0),
            "input_size": self.model.lstm.input_size,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
        }, self.config.lstm_model_path)
        self.logger.log_info(f"LSTM model saved to {self.config.lstm_model_path}")

    def load_model(self, df: Optional[pd.DataFrame] = None):
        """Load model from disk."""
        if not os.path.exists(self.config.lstm_model_path):
            self.logger.log_info("No saved LSTM model found")
            return

        checkpoint = torch.load(self.config.lstm_model_path, map_location=self.device,
                                weights_only=False)

        self.model = LSTMPriceModel(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore scaler
        scaler_restored = self._restore_scaler_from_checkpoint(checkpoint)
        if not scaler_restored and df is not None:
            self._fit_scaler_from_df(df)

        self.logger.log_info("LSTM model loaded")
