"""
train_mlp.py
Baseline surrogate: 6 design parameters -> (max von Mises stress, max displacement).

Run (from the folder that contains dataset2/):
    python train_mlp.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SEED = 0
MAX_EPOCHS = 1000
PATIENCE = 100          # early stopping on validation loss
BATCH = 32
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)
    
# ----------------------------- data -----------------------------
d = np.load("dataset2/bracket_dataset.npz")
X = d["params"].astype(np.float32)                                   # L, H, w, g, r, theta
Y = np.stack([d["vm_max"], d["u_max"]], axis=1).astype(np.float32)   # stress (MPa), disp (mm)
t_fea = float(d["t_fea"].mean())                                     # mean ANSYS time per sample, s
n = len(X)

perm = np.random.default_rng(SEED).permutation(n)
n_tr, n_va = int(0.70 * n), int(0.15 * n)
tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
print(f"samples: {n}  train {len(tr)}  val {len(va)}  test {len(te)}")

# inputs: z-score; targets: log, then z-score (stats from the training set only)
x_mu, x_sd = X[tr].mean(0), X[tr].std(0)
Ylog = np.log(Y)
y_mu, y_sd = Ylog[tr].mean(0), Ylog[tr].std(0)


def norm_x(a):
    return (a - x_mu) / x_sd


def norm_y(a):
    return (np.log(a) - y_mu) / y_sd


def denorm_y(z):
    return np.exp(z * y_sd + y_mu)


def tt(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


Xtr, Ytr = tt(norm_x(X[tr])), tt(norm_y(Y[tr]))
Xva, Yva = tt(norm_x(X[va])), tt(norm_y(Y[va]))
Xte = tt(norm_x(X[te]))

# ----------------------------- model -----------------------------
model = nn.Sequential(
    nn.Linear(6, 64), nn.SiLU(),
    nn.Linear(64, 64), nn.SiLU(),
    nn.Linear(64, 64), nn.SiLU(),
    nn.Linear(64, 2),
)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ----------------------------- training -----------------------------
best, bad, best_state = float("inf"), 0, None
hist = []
t0 = time.perf_counter()
for ep in range(MAX_EPOCHS):
    model.train()
    order = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), BATCH):
        idx = order[i:i + BATCH]
        opt.zero_grad()
        loss_fn(model(Xtr[idx]), Ytr[idx]).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tl = loss_fn(model(Xtr), Ytr).item()
        vl = loss_fn(model(Xva), Yva).item()
    hist.append((tl, vl))

    if vl < best - 1e-6:
        best, bad = vl, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    if ep % 50 == 0:
        print(f"epoch {ep:4d}  train {tl:.5f}  val {vl:.5f}")
    if bad >= PATIENCE:
        print(f"early stop at epoch {ep} (best val {best:.5f})")
        break

model.load_state_dict(best_state)
print(f"training time: {time.perf_counter() - t0:.1f} s, epochs run: {len(hist)}")

# ----------------------------- evaluation (test set) -----------------------------
model.eval()
with torch.no_grad():
    pred = denorm_y(model(Xte).numpy())
true = Y[te]
rel = np.abs(pred - true) / true * 100.0
ss_res = ((pred - true) ** 2).sum(0)
ss_tot = ((true - true.mean(0)) ** 2).sum(0)
r2 = 1 - ss_res / ss_tot

names = ["max von Mises (MPa)", "max displacement (mm)"]
print("\nTEST SET RESULTS")
for k, nm in enumerate(names):
    print(f"{nm:24s} R2 = {r2[k]:.4f} | rel. error mean {rel[:, k].mean():.2f}%  "
          f"median {np.median(rel[:, k]):.2f}%  90th pct {np.percentile(rel[:, k], 90):.2f}%")

# ----------------------------- speed-up -----------------------------
x1 = Xte[:1]
with torch.no_grad():
    for _ in range(50):          # warm-up
        model(x1)
    t0 = time.perf_counter()
    for _ in range(1000):
        model(x1)
    t_inf = (time.perf_counter() - t0) / 1000
print(f"\nANSYS per sample (build+mesh+solve+post): {t_fea:.2f} s")
print(f"MLP per sample: {t_inf * 1e3:.3f} ms  ->  speed-up about {t_fea / t_inf:,.0f}x")

# ----------------------------- plots + save -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for k, nm in enumerate(names):
    lo, hi = true[:, k].min(), true[:, k].max()
    ax[k].scatter(true[:, k], pred[:, k], s=12)
    ax[k].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[k].set_xlabel("ANSYS " + nm)
    ax[k].set_ylabel("MLP " + nm)
    ax[k].set_title(f"{nm}: R2 = {r2[k]:.3f}")
h = np.array(hist)
ax[2].semilogy(h[:, 0], label="train")
ax[2].semilogy(h[:, 1], label="validation")
ax[2].set_xlabel("epoch")
ax[2].set_ylabel("MSE (normalised log targets)")
ax[2].legend()
plt.tight_layout()
plt.savefig("mlp_results.png", dpi=130)
plt.show()

torch.save({"state": model.state_dict(), "x_mu": x_mu, "x_sd": x_sd,
            "y_mu": y_mu, "y_sd": y_sd, "test_idx": te}, "mlp_baseline.pt")
print("saved mlp_baseline.pt and mlp_results.png")  