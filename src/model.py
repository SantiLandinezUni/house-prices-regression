"""The PyTorch network and its scikit-learn wrapper."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class TabularMLP(nn.Module):
    """Fully-connected network for tabular regression.

    Each hidden block is Linear -> BatchNorm -> ReLU -> Dropout. BatchNorm keeps activations
    on a stable scale between layers; dropout is the main defence against memorising roughly
    1,450 training rows with several hundred input features.
    """

    def __init__(self, n_features, hidden=(128, 64), dropout=0.4):
        super().__init__()
        layers, previous = [], n_features
        for width in hidden:
            layers += [
                nn.Linear(previous, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn interface around `TabularMLP`, with early stopping.

    Implementing fit/predict lets the network sit inside a `Pipeline`, so cross-validation
    refits the preprocessing and the network together inside each fold. Writing a bare
    training loop over a pre-transformed matrix would reintroduce leakage.

    The defaults are the configuration chosen in section 11 of the notebook.
    """

    def __init__(self, hidden=(128, 64), dropout=0.4, lr=1e-3, weight_decay=1e-4,
                 batch_size=64, max_epochs=300, patience=25, val_fraction=0.15,
                 random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Standardise the target: log-prices sit near 12.0 with sd 0.4, which converges
        # slowly. Computed from training rows only, so this cannot leak.
        self.y_mean_, self.y_std_ = y.mean(), y.std()
        y_scaled = (y - self.y_mean_) / self.y_std_

        # Early stopping needs a validation signal, and it cannot come from the outer CV
        # fold -- that is the held-out data. So carve it out of the training rows instead.
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_fraction))
        val_idx, train_idx = order[:n_val], order[n_val:]

        X_tr, y_tr = torch.tensor(X[train_idx]), torch.tensor(y_scaled[train_idx])
        X_va, y_va = torch.tensor(X[val_idx]), torch.tensor(y_scaled[val_idx])

        self.model_ = TabularMLP(X.shape[1], self.hidden, self.dropout)
        optimiser = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=8)
        criterion = nn.MSELoss()

        best_loss, best_state, stalled = float("inf"), None, 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            shuffled = torch.randperm(len(X_tr))
            for start in range(0, len(X_tr), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                if len(batch) < 2:            # BatchNorm1d needs at least two rows
                    continue
                optimiser.zero_grad()
                criterion(self.model_(X_tr[batch]), y_tr[batch]).backward()
                optimiser.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_va), y_va).item()
            scheduler.step(val_loss)

            if val_loss < best_loss - 1e-5:
                best_loss, stalled = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                stalled += 1
                if stalled >= self.patience:
                    break

        self.best_epoch_ = epoch
        if best_state is not None:
            self.model_.load_state_dict(best_state)   # keep the best epoch, not the last
        return self

    def predict(self, X):
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            scaled = self.model_(X).numpy()
        return scaled * self.y_std_ + self.y_mean_    # back to the log-price scale
