AI Structural Analysis Dashboard
================================

1. Install Python 3.10-3.12 (recommended for PyTorch compatibility).
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended):

   Windows:
     py -m venv .venv
     .venv\Scripts\activate

4. Install packages:
     python -m pip install -r requirements.txt

5. Launch:
     python -m streamlit run app.py

The browser will open the dashboard. Change the six design parameters in the
sidebar, inspect the live bracket drawing, and click "Run both models".

Files required beside app.py:
- mlp_baseline.pt
- unet_baseline.pt

The dashboard does not need dataset2/bracket_dataset.npz for inference because
the trained checkpoints contain the required model weights and MLP scaling
statistics. The U-Net input mask is generated from the same bracket outline and
128x128, 0..120 mm image domain used by the dataset generator.

Scope:
This dashboard is a proof-of-concept for the trained 2D bracket family and the
training parameter ranges only. Load magnitude is fixed at 1000 N. It is not a
replacement for validated ANSYS analysis or engineering sign-off.
