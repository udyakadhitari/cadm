import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

# ---------------------------- configuration ----------------------------
ROOT = Path(__file__).resolve().parent
MLP_PATH = ROOT / "mlp_baseline.pt"
UNET_PATH = ROOT / "unet_baseline.pt"
GRID = 128
DOMAIN = 120.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="AI Structural Analysis", page_icon="🏗️", layout="wide")

# ---------------------------- exact model architectures ----------------------------
@st.cache_resource
def load_models():
    mlp_ckpt = torch.load(MLP_PATH, map_location="cpu", weights_only=False)
    mlp = nn.Sequential(
        nn.Linear(6, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 2),
    )
    mlp.load_state_dict(mlp_ckpt["state"])
    mlp.eval()

    def Block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self, cin=4, b=16):
            super().__init__()
            self.e1, self.e2 = Block(cin, b), Block(b, 2*b)
            self.e3, self.e4 = Block(2*b, 4*b), Block(4*b, 8*b)
            self.bott = Block(8*b, 16*b)
            self.up4, self.d4 = nn.ConvTranspose2d(16*b, 8*b, 2, stride=2), Block(16*b, 8*b)
            self.up3, self.d3 = nn.ConvTranspose2d(8*b, 4*b, 2, stride=2), Block(8*b, 4*b)
            self.up2, self.d2 = nn.ConvTranspose2d(4*b, 2*b, 2, stride=2), Block(4*b, 2*b)
            self.up1, self.d1 = nn.ConvTranspose2d(2*b, b, 2, stride=2), Block(2*b, b)
            self.out = nn.Conv2d(b, 1, 1)
        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool2d(e1, 2))
            e3 = self.e3(F.max_pool2d(e2, 2))
            e4 = self.e4(F.max_pool2d(e3, 2))
            b = self.bott(F.max_pool2d(e4, 2))
            d4 = self.d4(torch.cat([self.up4(b), e4], 1))
            d3 = self.d3(torch.cat([self.up3(d4), e3], 1))
            d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.out(d1)

    unet_ckpt = torch.load(UNET_PATH, map_location=DEVICE, weights_only=False)
    unet = UNet().to(DEVICE)
    unet.load_state_dict(unet_ckpt["state"])
    unet.eval()
    return mlp, mlp_ckpt, unet, unet_ckpt

# ---------------------------- geometry copied from dataset generator ----------------------------
TAN22 = np.tan(np.deg2rad(22.5))
SQ2 = np.sqrt(2.0)

def fillet_geometry(w, g, r):
    t = r * TAN22
    if 2*t >= g*SQ2:
        raise ValueError("Fillet radius is too large for this gusset size.")
    px, py = w + g, w
    qx, qy = w, w + g
    T1 = (px+t, w)
    T2 = (px-t/SQ2, w+t/SQ2)
    C1 = (px+t, w+r)
    T3 = (w+t/SQ2, qy-t/SQ2)
    T4 = (w, qy+t)
    C2 = (w+r, qy+t)
    return T1, T2, C1, T3, T4, C2

def arc_points(c, a, b, n=24):
    a0 = np.arctan2(a[1]-c[1], a[0]-c[0])
    b0 = np.arctan2(b[1]-c[1], b[0]-c[0])
    d = (b0-a0+np.pi) % (2*np.pi)-np.pi
    ang = a0+d*np.linspace(0, 1, n)
    rad = np.hypot(a[0]-c[0], a[1]-c[1])
    return [(c[0]+rad*np.cos(a), c[1]+rad*np.sin(a)) for a in ang]

def bracket_outline(L, H, w, g, r):
    T1, T2, C1, T3, T4, C2 = fillet_geometry(w, g, r)
    pts = [(0, 0), (L, 0), (L, w)]
    pts += arc_points(C1, T1, T2)
    pts += arc_points(C2, T3, T4)
    pts += [(w, H), (0, H)]
    return pts

def make_mask(points):
    # Vectorised point-in-polygon implementation; same 0..120 mm image domain as training.
    from matplotlib.path import Path as MplPath
    xs = np.linspace(0, DOMAIN, GRID)
    ys = np.linspace(0, DOMAIN, GRID)
    X, Y = np.meshgrid(xs, ys)
    inside = MplPath(points).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(GRID, GRID)
    return inside.astype(np.float32), xs, ys

def dimension_figure(points, L, H, w, g, r, theta):
    x, y = zip(*(points + [points[0]]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself",
                             fillcolor="rgba(75,145,210,0.20)", line=dict(color="#4B91D2", width=3),
                             name="Bracket"))
    # fixed wall
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, H], mode="lines", line=dict(color="#333", width=7),
                             name="Fixed wall"))
    # load arrow at tip face; direction convention from training script
    th = np.deg2rad(theta)
    dx, dy = np.sin(th), -np.cos(th)
    fig.add_annotation(x=L+dx*max(12, L*0.13), y=w/2+dy*max(12, L*0.13),
                       ax=L, ay=w/2, xref="x", yref="y", axref="x", ayref="y",
                       text="1000 N", showarrow=True, arrowhead=3, arrowsize=1.4,
                       arrowwidth=3, arrowcolor="#D64545", font=dict(color="#D64545"))
    # dimension extension lines and labels
    fig.add_annotation(x=L/2, y=-7, text=f"L = {L:.0f} mm", showarrow=False)
    fig.add_annotation(x=-8, y=H/2, text=f"H = {H:.0f} mm", textangle=-90, showarrow=False)
    fig.add_annotation(x=L+2, y=w/2, text=f"w = {w:.1f} mm", showarrow=False, xanchor="left")
    fig.add_annotation(x=w+g/2, y=w+g/2+5, text=f"g = {g:.1f} mm", showarrow=False)
    fig.add_annotation(x=w+g+3, y=w-4, text=f"r = {r:.1f} mm", showarrow=False, xanchor="left")
    fig.update_layout(height=440, margin=dict(l=30,r=30,t=20,b=35),
                      xaxis=dict(title="x (mm)", range=[-15, 135], scaleanchor="y", scaleratio=1, zeroline=False),
                      yaxis=dict(title="y (mm)", range=[-12, 135], zeroline=False),
                      showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    return fig

# ---------------------------- UI ----------------------------
st.title("AI-Powered Structural Analysis")
st.caption("Deep-learning surrogate models for a 2D gusset-supported L-bracket")

with st.sidebar:
    st.header("Design parameters")
    L = st.slider("Arm length, L (mm)", 80.0, 120.0, 100.0, 1.0)
    H = st.slider("Wall height, H (mm)", 80.0, 120.0, 100.0, 1.0)
    w = st.slider("Arm / leg width, w (mm)", 10.0, 25.0, 17.5, 0.5)
    g = st.slider("Gusset size, g (mm)", 5.0, 40.0, 22.5, 0.5)
    r = st.slider("Fillet radius, r (mm)", 2.0, 6.0, 4.0, 0.25)
    theta = st.slider("Load angle, θ (degrees)", -45.0, 45.0, 0.0, 1.0)
    st.divider()
    st.caption("Fixed project conditions: 1000 N load, steel, E = 200 GPa, ν = 0.3, 10 mm thickness, linear static analysis.")

try:
    pts = bracket_outline(L, H, w, g, r)
    mask, xs, ys = make_mask(pts)
except Exception as e:
    st.error(f"Invalid geometry: {e}")
    st.stop()

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Live bracket geometry")
    st.plotly_chart(dimension_figure(pts, L, H, w, g, r, theta), use_container_width=True, config={"displayModeBar": False})
    st.caption("Geometry updates automatically when a dimension or load angle changes.")
with right:
    st.subheader("AI predictions")
    st.write("Run both trained models for the current design.")
    run = st.button("▶  Run both models", type="primary", use_container_width=True)
    if run:
        if not MLP_PATH.exists() or not UNET_PATH.exists():
            st.error("Model checkpoint files are missing. Keep both .pt files beside app.py.")
            st.stop()
        mlp, mc, unet, uc = load_models()
        params = np.array([[L, H, w, g, r, theta]], dtype=np.float32)
        x_mu = mc["x_mu"].detach().cpu().numpy() if torch.is_tensor(mc["x_mu"]) else np.asarray(mc["x_mu"])
        x_sd = mc["x_sd"].detach().cpu().numpy() if torch.is_tensor(mc["x_sd"]) else np.asarray(mc["x_sd"])
        xnorm = (params - x_mu) / x_sd
        t0 = time.perf_counter()
        with torch.inference_mode():
            yhat_z = mlp(torch.tensor(xnorm, dtype=torch.float32)).numpy()[0]
        y_sd = mc["y_sd"].detach().cpu().numpy() if torch.is_tensor(mc["y_sd"]) else np.asarray(mc["y_sd"])
        y_mu = mc["y_mu"].detach().cpu().numpy() if torch.is_tensor(mc["y_mu"]) else np.asarray(mc["y_mu"])
        mlp_out = np.exp(yhat_z * y_sd + y_mu)
        mlp_ms = (time.perf_counter() - t0) * 1000

        # U-Net inputs must match the training pipeline: mask, theta/45, x-grid, y-grid.
        yy, xx = np.meshgrid(np.linspace(0, 1, GRID), np.linspace(0, 1, GRID), indexing="ij")
        th_channel = np.full((GRID, GRID), theta/45.0, dtype=np.float32)
        xin = np.stack([mask, th_channel, xx.astype(np.float32), yy.astype(np.float32)], axis=0)[None]
        xt = torch.tensor(xin, dtype=torch.float32, device=DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            zmap = unet(xt)[:, 0].cpu().numpy()[0]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        unet_ms = (time.perf_counter() - t1) * 1000
        zmap = np.clip(zmap, 0, None)
        stress_map = (np.expm1(zmap * float(uc["T_MAX"])) * float(uc["C0"])) * mask
        stress_map = np.maximum(stress_map, 0)

        st.session_state["prediction"] = {
            "stress": float(mlp_out[0]), "disp": float(mlp_out[1]),
            "mlp_ms": mlp_ms, "unet_ms": unet_ms,
            "stress_map": stress_map, "params": params[0].tolist()
        }

    pred = st.session_state.get("prediction")
    if pred and pred["params"] == [L, H, w, g, r, theta]:
        a, b = st.columns(2)
        a.metric("Maximum von Mises stress", f"{pred['stress']:.2f} MPa")
        b.metric("Maximum displacement", f"{pred['disp']:.4f} mm")
        a, b = st.columns(2)
        a.metric("MLP inference", f"{pred['mlp_ms']:.2f} ms")
        b.metric("U-Net inference", f"{pred['unet_ms']:.2f} ms")
        st.caption("These are surrogate predictions, not a fresh ANSYS solution.")
    elif pred:
        st.info("Design changed. Click **Run both models** to update predictions.")
    else:
        st.info("Set the dimensions, then click **Run both models**.")

st.divider()
st.subheader("Predicted von Mises stress distribution")
pred = st.session_state.get("prediction")
if pred and pred["params"] == [L, H, w, g, r, theta]:
    fig = go.Figure(go.Heatmap(z=pred["stress_map"], x=xs, y=ys, colorscale="Jet",
                               colorbar=dict(title="MPa"), zmin=0))
    fig.add_trace(go.Scatter(x=[p[0] for p in pts+[pts[0]]], y=[p[1] for p in pts+[pts[0]]],
                             mode="lines", line=dict(color="white", width=1.2), name="Bracket boundary"))
    fig.update_layout(height=560, margin=dict(l=20,r=20,t=20,b=20),
                      xaxis=dict(title="x (mm)", scaleanchor="y", scaleratio=1),
                      yaxis=dict(title="y (mm)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("U-Net output is a 128 × 128 predicted stress field. The model's reported test-set mean relative L2 error was about 6%; peak hotspots were underestimated on average, so verify critical designs with FEA.")
else:
    st.info("The stress map will appear here after running the U-Net.")

with st.expander("Model scope and engineering limitations"):
    st.markdown("""
    - Geometry: the same 2D gusset-supported L-bracket family used for training.
    - Valid input ranges: L and H 80–120 mm; w 10–25 mm; g 5–40 mm; r 2–6 mm; θ −45° to +45°.
    - Fixed conditions: 1000 N total load, steel properties and 10 mm thickness as used in the training simulations.
    - Analysis represented: linear-elastic 2D plane-stress simulation.
    - The models were tested on unseen designs inside the training ranges (interpolation), not on other component geometries or extrapolated designs.
    - Use ANSYS/validated FEA for final engineering decisions, especially near stress hotspots; the U-Net can underestimate peak stress.
    """)
