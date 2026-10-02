"""PyTorch LSTM Quantile Time-Series Forecaster for Multi-Horizon Price Fan Charts."""

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def set_torch_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QuantileLoss(nn.Module):
    """Pinball loss for multi-quantile regression (e.g. p10, p50, p90)."""

    def __init__(self, quantiles: list[float] | None = None):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (batch_size, num_quantiles), target: (batch_size, 1)
        losses = []
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i : i + 1]
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss)
        return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


class LSTMPriceForecaster(nn.Module):
    """Recurrent neural network with quantile linear output for probabilistic forecasting."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 3,  # 3 quantiles (p10, p50, p90)
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        # Ensure quantile ordering p10 <= p50 <= p90
        p10 = out[:, 0:1]
        p50 = p10 + torch.relu(out[:, 1:2])
        p90 = p50 + torch.relu(out[:, 2:3])
        return torch.cat([p10, p50, p90], dim=1)


class LSTMPipeline:
    """Manages data preparation, rolling sequence windowing, training, and inference."""

    def __init__(
        self,
        seq_length: int = 30,
        forecast_horizon: int = 21,
        hidden_dim: int = 64,
        num_layers: int = 2,
        quantiles: list[float] | None = None,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 35,
        patience: int = 8,
        seed: int = 42,
    ):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_torch_seed(seed)

        self.model = None
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
        self.feature_cols = ["Close", "Volume", "ret_1d", "vol_21d", "rsi_14"]
        self.metrics = {}

    def _create_sequences(
        self,
        df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Creates sliding sequence windows without look-ahead leakage."""
        data = df.copy().reset_index(drop=True)
        for col in self.feature_cols:
            if col not in data.columns:
                if col == "ret_1d":
                    data[col] = data["Close"].pct_change().fillna(0)
                elif col == "vol_21d":
                    data[col] = data["ret_1d"].rolling(21, min_periods=1).std().fillna(0)
                elif col == "rsi_14":
                    data[col] = 50.0

        feat_matrix = data[self.feature_cols].fillna(0).values
        close_vals = data["Close"].values

        # Normalize features
        self.scaler_mean = np.mean(feat_matrix, axis=0)
        self.scaler_std = np.std(feat_matrix, axis=0) + 1e-8
        norm_matrix = (feat_matrix - self.scaler_mean) / self.scaler_std

        X_seq, y_target = [], []
        n_samples = len(data) - self.seq_length - self.forecast_horizon + 1

        for i in range(max(0, n_samples)):
            X_seq.append(norm_matrix[i : i + self.seq_length])
            # Target: Log return over the forecast horizon
            curr_close = close_vals[i + self.seq_length - 1]
            future_close = close_vals[i + self.seq_length - 1 + self.forecast_horizon]
            ret = (future_close - curr_close) / curr_close
            y_target.append(ret)

        if not X_seq:
            return torch.empty(0), torch.empty(0)

        X_t = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_t = torch.tensor(np.array(y_target), dtype=torch.float32).unsqueeze(1)
        return X_t, y_t

    def train_walk_forward(self, df: pd.DataFrame) -> dict[str, Any]:
        """Trains LSTM using walk-forward chronological train/val split."""
        X_all, y_all = self._create_sequences(df)
        if len(X_all) < 50:
            return {"status": "insufficient_data"}

        split_idx = int(len(X_all) * 0.8)
        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_val, y_val = X_all[split_idx:], y_all[split_idx:]

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = LSTMPriceForecaster(
            input_dim=len(self.feature_cols),
            hidden_dim=48,
            num_layers=2,
            output_dim=len(self.quantiles),
        ).to(self.device)

        criterion = QuantileLoss(quantiles=self.quantiles)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val.to(self.device))
                val_loss = criterion(val_preds, y_val.to(self.device)).item()

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Evaluation metrics on validation set
        self.model.eval()
        with torch.no_grad():
            preds_val = self.model(X_val.to(self.device)).cpu().numpy()
            y_val_np = y_val.numpy().flatten()

        p10 = preds_val[:, 0]
        p50 = preds_val[:, 1]
        p90 = preds_val[:, 2]

        mae = float(np.mean(np.abs(y_val_np - p50)))
        rmse = float(np.sqrt(np.mean((y_val_np - p50) ** 2)))
        mape = float(np.mean(np.abs((y_val_np - p50) / (np.abs(y_val_np) + 1e-4))) * 100)
        coverage = float(np.mean((y_val_np >= p10) & (y_val_np <= p90)) * 100)  # Should be ~80%
        interval_width = float(np.mean(p90 - p10))
        directional_acc = float(np.mean((p50 > 0) == (y_val_np > 0)) * 100)

        self.metrics = {
            "model_name": "PyTorch LSTM Quantile",
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape_pct": round(min(100.0, mape), 2),
            "interval_coverage_80pct": round(coverage, 1),
            "mean_interval_width": round(interval_width, 4),
            "directional_accuracy_pct": round(directional_acc, 2),
            "device": str(self.device),
        }
        return self.metrics

    def predict_future_fan(
        self,
        df: pd.DataFrame,
        horizon_days: int = 63,
    ) -> list[dict[str, Any]]:
        """Generates multi-step forward price fan chart paths (p10, p50, p90)."""
        if self.model is None:
            # Generate fallback fan chart based on historical drift and volatility
            latest_price = float(df["Close"].iloc[-1])
            daily_vol = float(df["Close"].pct_change().std())
            fan_points = []
            for d in range(1, horizon_days + 1):
                drift = 0.0004 * d
                spread = 1.645 * daily_vol * np.sqrt(d)  # 90% confidence
                p50 = latest_price * (1 + drift)
                p10 = latest_price * (1 + drift - spread)
                p90 = latest_price * (1 + drift + spread)
                fan_points.append(
                    {
                        "day": d,
                        "price_p10": round(float(p10), 2),
                        "price_p50": round(float(p50), 2),
                        "price_p90": round(float(p90), 2),
                    }
                )
            return fan_points

        self.model.eval()
        data = df.copy().reset_index(drop=True)
        for col in self.feature_cols:
            if col not in data.columns:
                if col == "ret_1d":
                    data[col] = data["Close"].pct_change().fillna(0)
                elif col == "vol_21d":
                    data[col] = data["ret_1d"].rolling(21, min_periods=1).std().fillna(0)
                elif col == "rsi_14":
                    data[col] = 50.0

        latest_seq = data[self.feature_cols].tail(self.seq_length).fillna(0).values
        norm_seq = (latest_seq - self.scaler_mean) / self.scaler_std
        x_tensor = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(x_tensor).cpu().numpy()[0]

        latest_price = float(df["Close"].iloc[-1])
        ret_p10, ret_p50, ret_p90 = preds[0], preds[1], preds[2]

        fan_points = []
        for d in range(1, horizon_days + 1):
            fraction = d / self.forecast_horizon
            # Interpolate fan paths
            p10 = latest_price * (1 + ret_p10 * fraction)
            p50 = latest_price * (1 + ret_p50 * fraction)
            p90 = latest_price * (1 + ret_p90 * fraction)
            fan_points.append(
                {
                    "day": d,
                    "price_p10": round(float(p10), 2),
                    "price_p50": round(float(p50), 2),
                    "price_p90": round(float(p90), 2),
                }
            )

        return fan_points
