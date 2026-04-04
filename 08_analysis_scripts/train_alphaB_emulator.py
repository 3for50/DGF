#!/usr/bin/env python3
"""
Train PCA + NN emulator for C_l(H0, omega_cdm, logA, alpha_B_scale).
Uses the training data from generate_alphaB_training.py.

Architecture: PCA decomposition of C_l, then NN mapping params -> PCA coefficients.
"""
import numpy as np
import json, os, time

DATADIR = "/home/joe-research/dgf_training/cl_emulator_alphaB"
MODELDIR = os.path.join(DATADIR, "model")
os.makedirs(MODELDIR, exist_ok=True)

# ── Load training data ──
print("Loading training data...")
params = np.load(os.path.join(DATADIR, "params.npy"))
cl_tt = np.load(os.path.join(DATADIR, "cl_tt.npy"))
cl_ee = np.load(os.path.join(DATADIR, "cl_ee.npy"))
cl_te = np.load(os.path.join(DATADIR, "cl_te.npy"))
ell = np.load(os.path.join(DATADIR, "ell.npy"))

with open(os.path.join(DATADIR, "metadata.json")) as f:
    meta = json.load(f)

n_samples, n_ell = cl_tt.shape
n_params = params.shape[1]
print(f"  {n_samples} samples, {n_ell} multipoles, {n_params} parameters")

# ── Normalise parameters ──
param_mean = params.mean(axis=0)
param_std = params.std(axis=0)
params_norm = (params - param_mean) / param_std

# ── Log-transform C_l for better PCA ──
# TT and EE are positive; TE can be negative
log_cl_tt = np.log(cl_tt + 1e-30)
log_cl_ee = np.log(cl_ee + 1e-30)
# For TE: use the raw values (can be negative)

# ── PCA decomposition ──
N_PCA = 30  # number of PCA components

def fit_pca(X, n_components):
    """Simple PCA via SVD."""
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    coefficients = X_centered @ components.T
    # Reconstruction error
    X_recon = coefficients @ components + mean
    rel_err = np.abs(X - X_recon) / (np.abs(X) + 1e-30)
    print(f"  PCA {n_components} components: mean rel error = {rel_err.mean():.6f}, "
          f"max = {rel_err.max():.6f}")
    return mean, components, coefficients

print("\nFitting PCA...")
tt_mean, tt_components, tt_coeffs = fit_pca(log_cl_tt, N_PCA)
ee_mean, ee_components, ee_coeffs = fit_pca(log_cl_ee, N_PCA)
te_mean, te_components, te_coeffs = fit_pca(cl_te, N_PCA)

# ── Train neural networks: params -> PCA coefficients ──
print("\nTraining neural networks...")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    USE_TORCH = True
    print("  Using PyTorch")
except ImportError:
    USE_TORCH = False

if not USE_TORCH:
    try:
        from sklearn.neural_network import MLPRegressor
        USE_SKLEARN = True
        print("  Using sklearn")
    except ImportError:
        USE_SKLEARN = False
        print("  WARNING: No ML framework available, using linear regression")

# Split train/validation (90/10)
n_val = max(1, n_samples // 10)
idx = np.random.RandomState(42).permutation(n_samples)
train_idx, val_idx = idx[n_val:], idx[:n_val]

if USE_TORCH:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    class PCANet(nn.Module):
        def __init__(self, n_in, n_out, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, n_out),
            )
        def forward(self, x):
            return self.net(x)

    def train_net(X_train, Y_train, X_val, Y_val, label, epochs=2000, lr=1e-3):
        n_out = Y_train.shape[1]
        model = PCANet(n_params, n_out).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_v = torch.tensor(Y_val, dtype=torch.float32).to(device)

        best_val = float('inf')
        best_state = None

        for ep in range(epochs):
            model.train()
            pred = model(X_t)
            loss = ((pred - Y_t)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (ep + 1) % 200 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_v)
                    val_loss = ((val_pred - Y_v)**2).mean().item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"    [{label}] ep {ep+1}: train={loss.item():.6f}, val={val_loss:.6f}")

        model.load_state_dict(best_state)
        return model

    # Normalise PCA coefficients
    tt_coeff_mean = tt_coeffs[train_idx].mean(axis=0)
    tt_coeff_std = tt_coeffs[train_idx].std(axis=0) + 1e-10
    ee_coeff_mean = ee_coeffs[train_idx].mean(axis=0)
    ee_coeff_std = ee_coeffs[train_idx].std(axis=0) + 1e-10
    te_coeff_mean = te_coeffs[train_idx].mean(axis=0)
    te_coeff_std = te_coeffs[train_idx].std(axis=0) + 1e-10

    tt_coeffs_norm = (tt_coeffs - tt_coeff_mean) / tt_coeff_std
    ee_coeffs_norm = (ee_coeffs - ee_coeff_mean) / ee_coeff_std
    te_coeffs_norm = (te_coeffs - te_coeff_mean) / te_coeff_std

    print("  Training TT network...")
    tt_model = train_net(params_norm[train_idx], tt_coeffs_norm[train_idx],
                         params_norm[val_idx], tt_coeffs_norm[val_idx], "TT")
    print("  Training EE network...")
    ee_model = train_net(params_norm[train_idx], ee_coeffs_norm[train_idx],
                         params_norm[val_idx], ee_coeffs_norm[val_idx], "EE")
    print("  Training TE network...")
    te_model = train_net(params_norm[train_idx], te_coeffs_norm[train_idx],
                         params_norm[val_idx], te_coeffs_norm[val_idx], "TE")

    # Save models
    torch.save(tt_model.state_dict(), os.path.join(MODELDIR, "tt_net.pt"))
    torch.save(ee_model.state_dict(), os.path.join(MODELDIR, "ee_net.pt"))
    torch.save(te_model.state_dict(), os.path.join(MODELDIR, "te_net.pt"))

    # ── Validation ──
    print("\nValidation on held-out set...")
    tt_model.eval(); ee_model.eval(); te_model.eval()
    with torch.no_grad():
        X_v = torch.tensor(params_norm[val_idx], dtype=torch.float32).to(device)

        tt_pred_norm = tt_model(X_v).cpu().numpy()
        ee_pred_norm = ee_model(X_v).cpu().numpy()
        te_pred_norm = te_model(X_v).cpu().numpy()

    tt_pred_coeffs = tt_pred_norm * tt_coeff_std + tt_coeff_mean
    ee_pred_coeffs = ee_pred_norm * ee_coeff_std + ee_coeff_mean
    te_pred_coeffs = te_pred_norm * te_coeff_std + te_coeff_mean

    tt_pred = np.exp(tt_pred_coeffs @ tt_components + tt_mean)
    ee_pred = np.exp(ee_pred_coeffs @ ee_components + ee_mean)
    te_pred = te_pred_coeffs @ te_components + te_mean

    tt_true = cl_tt[val_idx]
    ee_true = cl_ee[val_idx]
    te_true = cl_te[val_idx]

    tt_err = np.abs(tt_pred - tt_true) / (np.abs(tt_true) + 1e-30)
    ee_err = np.abs(ee_pred - ee_true) / (np.abs(ee_true) + 1e-30)

    print(f"  TT: mean rel error = {tt_err.mean():.4%}, max = {tt_err.max():.4%}")
    print(f"  EE: mean rel error = {ee_err.mean():.4%}, max = {ee_err.max():.4%}")

    # Save normalisation
    norm = {
        "param_mean": param_mean.tolist(),
        "param_std": param_std.tolist(),
        "tt_coeff_mean": tt_coeff_mean.tolist(),
        "tt_coeff_std": tt_coeff_std.tolist(),
        "ee_coeff_mean": ee_coeff_mean.tolist(),
        "ee_coeff_std": ee_coeff_std.tolist(),
        "te_coeff_mean": te_coeff_mean.tolist(),
        "te_coeff_std": te_coeff_std.tolist(),
        "n_pca": N_PCA,
        "n_params": n_params,
        "param_names": meta["param_names"],
        "tt_err_mean": float(tt_err.mean()),
        "ee_err_mean": float(ee_err.mean()),
    }

else:
    # Fallback: sklearn MLPRegressor or linear
    from sklearn.neural_network import MLPRegressor

    def train_sklearn(X_train, Y_train, X_val, Y_val, label):
        model = MLPRegressor(hidden_layer_sizes=(256, 256, 256),
                            activation='relu', max_iter=5000,
                            early_stopping=True, validation_fraction=0.1,
                            random_state=42, verbose=False)
        model.fit(X_train, Y_train)
        pred = model.predict(X_val)
        mse = ((pred - Y_val)**2).mean()
        print(f"    [{label}] val MSE = {mse:.6f}")
        return model

    import pickle

    print("  Training TT...")
    tt_model = train_sklearn(params_norm[train_idx], tt_coeffs[train_idx],
                             params_norm[val_idx], tt_coeffs[val_idx], "TT")
    print("  Training EE...")
    ee_model = train_sklearn(params_norm[train_idx], ee_coeffs[train_idx],
                             params_norm[val_idx], ee_coeffs[val_idx], "EE")
    print("  Training TE...")
    te_model = train_sklearn(params_norm[train_idx], te_coeffs[train_idx],
                             params_norm[val_idx], te_coeffs[val_idx], "TE")

    with open(os.path.join(MODELDIR, "tt_net.pkl"), 'wb') as f:
        pickle.dump(tt_model, f)
    with open(os.path.join(MODELDIR, "ee_net.pkl"), 'wb') as f:
        pickle.dump(ee_model, f)
    with open(os.path.join(MODELDIR, "te_net.pkl"), 'wb') as f:
        pickle.dump(te_model, f)

    norm = {
        "param_mean": param_mean.tolist(),
        "param_std": param_std.tolist(),
        "n_pca": N_PCA,
        "n_params": n_params,
        "param_names": meta["param_names"],
        "backend": "sklearn",
    }

# Save PCA components
np.save(os.path.join(MODELDIR, "tt_pca_mean.npy"), tt_mean)
np.save(os.path.join(MODELDIR, "tt_pca_components.npy"), tt_components)
np.save(os.path.join(MODELDIR, "ee_pca_mean.npy"), ee_mean)
np.save(os.path.join(MODELDIR, "ee_pca_components.npy"), ee_components)
np.save(os.path.join(MODELDIR, "te_pca_mean.npy"), te_mean)
np.save(os.path.join(MODELDIR, "te_pca_components.npy"), te_components)
np.save(os.path.join(MODELDIR, "ell.npy"), ell)

with open(os.path.join(MODELDIR, "normalisation.json"), 'w') as f:
    json.dump(norm, f, indent=2)

print(f"\nModel saved to {MODELDIR}/")
print("TRAINING COMPLETE.")
