# Glaucoma Detection (Streamlit App)

A Streamlit-based demo that visualizes glaucoma predictions on retinal fundus images. The app highlights model attention, displays confidence cards, and aggregates batch results.

## Features
- Bulk image upload with animated progress feedback
- Mock or real AI predictions (PyTorch ResNet50)
- GradCAM-inspired heatmaps to show where the model focuses
- Styled confidence cards and summary bar chart
- Downloadable plain-text report for batch predictions

## Project Structure
```
├── app.py                 # Streamlit application
├── model.py               # GlaucomaResNet (ResNet50 wrapper)
├── glaucoma_resnet.pth    # ⬇️ Place your trained weights here (ignored by git)
├── inspect_checkpoint.py  # Utility script to inspect checkpoint contents
├── glaucoma_env/          # Optional venv (ignored by git)
└── README.md
```

## Requirements
- Python 3.10+
- Streamlit
- PyTorch + Torchvision (CPU build is fine)
- OpenCV, Pillow, Matplotlib, NumPy

Create a `requirements.txt` similar to:
```
streamlit
torch
torchvision
opencv-python
pillow
matplotlib
numpy
```
Install packages:
```bash
pip install -r requirements.txt
```

## Model Weights
The repository ignores `*.pth` by default (`.gitignore`). To use the real model:
1. Obtain `glaucoma_resnet.pth` from your private storage.
2. Place it at the project root (same folder as `app.py`).
3. The app will load it automatically if the file exists; otherwise it falls back to mock predictions.

> **Tip:** Store the weights privately (e.g., encrypted cloud bucket or Streamlit Secrets download link). Do not commit the model if it contains proprietary data or violates dataset licenses.

## Running Locally
```bash
streamlit run app.py
```
Open the provided local URL and upload fundus images to view predictions.

## Deploying on Streamlit Cloud
1. Push this repo to GitHub (weights excluded).
2. On Streamlit Cloud, set the repo and branch, and ensure `requirements.txt` is present.
3. Provide the model via one of these options:
   - Download in `app.py` during startup using a secure URL stored in Streamlit Secrets.
   - Manually upload the weight file via Streamlit Cloud’s file manager (private apps only).
4. Redeploy—the app will use the real model if the file is found on the server.

## Notes
- `inspect_checkpoint.py` helps verify the saved state dict structure if you retrain models.
- Update `model.py` accordingly if you switch to a different architecture or number of classes.

Enjoy exploring glaucoma detection visualizations! 👁️📊
