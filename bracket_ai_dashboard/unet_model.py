"""
train_unet.py
U-Net surrogate: bracket geometry (mask) + load angle -> von Mises stress map (128 x 128).

Run (from the folder that contains dataset2/):
    python train_unet.py
Uses the same 70/15/15 split as train_mlp.py so both models are scored on the same test designs.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
MAX_EPOCHS = 300
PATIENCE = 30           # early stopping on validation loss
BATCH = 8
LR = 1e-3
BASE = 16               # channel width of the first U-Net level
W_HOT = 4.0             # extra loss weight on high-stress pixels (helps the hotspot)
C0 = 10.0               # MPa, soft-log scale for the stress transform

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ----------------------------- data -----------------------------
d = np.load("dataset2/bracket_dataset.npz")
mask = d["mask"].astype(np.float32)          # (N, H, W) bracket shape, 1 inside
vm = d["vm"].astype(np.float32)              # (N, H, W) stress map in MPa
theta = d["params"][:, 5].astype(np.float32) # load angle, degrees
t_fea = float(d["t_fea"].mean())
N, H, W = mask.shape

# same split as train_mlp.py
perm = np.random.default_rng(SEED).permutation(N)
n_tr, n_va = int(0.70 * N), int(0.15 * N)
tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
print(f"samples: {N}  train {len(tr)}  val {len(va)}  test {len(te)}")


def fwd(v):
    return np.log1p(v / C0)


def inv(z):
    return np.expm1(z) * C0


T_MAX = float(fwd(vm[tr].max()))             # scale so training targets lie in [0, 1]

yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")  # row = y, col = x


def build_inputs(idx):
    """4 channels: mask, load angle (constant), x coordinate, y coordinate."""
    m = mask[idx]
    th = (theta[idx] / 45.0)[:, None, None] * np.ones((1, H, W), dtype=np.float32)
    xc = np.broadcast_to(xx, m.shape)
    yc = np.broadcast_to(yy, m.shape)
    return np.stack([m, th, xc, yc], axis=1).astype(np.float32)


def build_targets(idx):
    return (fwd(vm[idx]) / T_MAX)[:, None].astype(np.float32)


def tt(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


Xtr, Ztr, Mtr = tt(build_inputs(tr)), tt(build_targets(tr)), tt(mask[tr][:, None])
Xva, Zva, Mva = tt(build_inputs(va)), tt(build_targets(va)), tt(mask[va][:, None])
Xte = tt(build_inputs(te))


# ----------------------------- model -----------------------------
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin=4, b=BASE):
        super().__init__()
        self.e1, self.e2 = block(cin, b), block(b, 2 * b)
        self.e3, self.e4 = block(2 * b, 4 * b), block(4 * b, 8 * b)
        self.bott = block(8 * b, 16 * b)
        self.up4, self.d4 = nn.ConvTranspose2d(16 * b, 8 * b, 2, stride=2), block(16 * b, 8 * b)
        self.up3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2), block(8 * b, 4 * b)
        self.up2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), block(4 * b, 2 * b)
        self.up1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), block(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):                      # x: (B, 4, 128, 128)
        e1 = self.e1(x)                        # 128
        e2 = self.e2(F.max_pool2d(e1, 2))      # 64
        e3 = self.e3(F.max_pool2d(e2, 2))      # 32
        e4 = self.e4(F.max_pool2d(e3, 2))      # 16
        b = self.bott(F.max_pool2d(e4, 2))     # 8
        d4 = self.d4(torch.cat([self.up4(b), e4], 1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


model = UNet().to(device)
print("parameters: %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)


def masked_loss(pred, z, m):
    """L1 inside the bracket only, with extra weight on high-stress pixels."""
    w = (1.0 + W_HOT * z) * m
    return (torch.abs(pred - z) * w).sum() / w.sum()


def eval_loss(X, Z, M):
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), 16):
            xb, zb, mb = X[i:i + 16].to(device), Z[i:i + 16].to(device), M[i:i + 16].to(device)
            tot += masked_loss(model(xb), zb, mb).item() * len(xb)
            cnt += len(xb)
    return tot / cnt


# ----------------------------- training -----------------------------
best, bad, best_state, hist = float("inf"), 0, None, []
t0 = time.perf_counter()
for ep in range(MAX_EPOCHS):
    model.train()
    order = torch.randperm(len(Xtr))
    run, cnt = 0.0, 0
    for i in range(0, len(Xtr), BATCH):
        idx = order[i:i + BATCH]
        xb, zb, mb = Xtr[idx].to(device), Ztr[idx].to(device), Mtr[idx].to(device)
        opt.zero_grad()
        loss = masked_loss(model(xb), zb, mb)
        loss.backward()
        opt.step()
        run += loss.item() * len(idx)
        cnt += len(idx)
    tl = run / cnt
    vl = eval_loss(Xva, Zva, Mva)
    sched.step(vl)
    hist.append((tl, vl))

    if vl < best - 1e-6:
        best, bad = vl, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    if ep % 5 == 0 or bad == 0:
        print(f"epoch {ep:3d}  train {tl:.5f}  val {vl:.5f}  lr {opt.param_groups[0]['lr']:.1e}  "
              f"elapsed {time.perf_counter() - t0:.0f}s")
    if bad >= PATIENCE:
        print(f"early stop at epoch {ep} (best val {best:.5f})")
        break

model.load_state_dict(best_state)
print(f"training time: {(time.perf_counter() - t0) / 60:.1f} min, epochs run: {len(hist)}")


# ----------------------------- evaluation (test set, in MPa) -----------------------------
def predict(X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 16):
            out.append(model(X[i:i + 16].to(device))[:, 0].cpu().numpy())
    z = np.clip(np.concatenate(out), 0, None)
    return inv(z * T_MAX)


pred = predict(Xte) * mask[te]               # zero outside the bracket
true = vm[te]
m = mask[te] > 0

mae, rel, peak_err = [], [], []
for p, t, mm in zip(pred, true, m):
    mae.append(np.abs(p[mm] - t[mm]).mean())
    rel.append(np.linalg.norm(p[mm] - t[mm]) / np.linalg.norm(t[mm]) * 100)
    peak_err.append((p[mm].max() - t[mm].max()) / t[mm].max() * 100)
mae, rel, peak_err = map(np.array, (mae, rel, peak_err))

print("\nTEST SET RESULTS (stress maps)")
print(f"pixel MAE inside bracket : mean {mae.mean():.2f} MPa   median {np.median(mae):.2f} MPa")
print(f"relative L2 error        : mean {rel.mean():.2f}%   median {np.median(rel):.2f}%   90th pct {np.percentile(rel, 90):.2f}%")
print(f"peak stress error (signed): mean {peak_err.mean():+.2f}%   mean |error| {np.abs(peak_err).mean():.2f}%")
print("(negative signed peak error = hotspot underestimated)")

# ----------------------------- speed -----------------------------
x1 = Xte[:1].to(device)
model.eval()
with torch.no_grad():
    for _ in range(5):
        model(x1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(50):
        model(x1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = (time.perf_counter() - t1) / 50
print(f"\nANSYS per sample: {t_fea:.2f} s | U-Net per sample: {t_inf * 1e3:.2f} ms ({device.type}) "
      f"-> speed-up about {t_fea / t_inf:,.0f}x")

# ----------------------------- plots -----------------------------
order = np.argsort(rel)
pick = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]   # best ... worst
labels = ["best", "typical-good", "typical-poor", "worst"]
fig, ax = plt.subplots(3, 4, figsize=(15, 11))
for j, i in enumerate(pick):
    vmax = true[i].max()
    ax[0, j].imshow(true[i], origin="lower", cmap="jet", vmin=0, vmax=vmax)
    ax[0, j].set_title(f"ANSYS ({labels[j]}), peak {true[i].max():.0f} MPa", fontsize=9)
    ax[1, j].imshow(pred[i], origin="lower", cmap="jet", vmin=0, vmax=vmax)
    ax[1, j].set_title(f"U-Net, peak {pred[i].max():.0f} MPa", fontsize=9)
    im = ax[2, j].imshow(np.abs(pred[i] - true[i]), origin="lower", cmap="magma", vmin=0, vmax=vmax * 0.3)
    ax[2, j].set_title(f"|error|, rel L2 {rel[i]:.1f}%", fontsize=9)
plt.tight_layout()
plt.savefig("unet_results.png", dpi=120)

plt.figure(figsize=(6, 4))
h = np.array(hist)
plt.semilogy(h[:, 0], label="train")
plt.semilogy(h[:, 1], label="validation")
plt.xlabel("epoch")
plt.ylabel("masked weighted L1 (normalised)")
plt.legend()
plt.tight_layout()
plt.savefig("unet_loss.png", dpi=120)
plt.show()

torch.save({"state": model.state_dict(), "T_MAX": T_MAX, "C0": C0, "test_idx": te}, "unet_baseline.pt")
print("saved unet_baseline.pt, unet_results.png, unet_loss.png")