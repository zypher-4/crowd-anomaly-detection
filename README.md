# Crowd Behavior Anomaly Detection

Unsupervised Video Behavior Anomaly Mining in Crowd Surveillance  
**CSE 572 — ASU, Spring 2025**

## Team
| Name | ASU ID | Role |
|------|--------|------|
| Wasil Ahmad | 1237583973 | Data Preprocessing & Optical Flow |
| Siddharth Jain | 1237112528 | Autoencoder Architecture |
| Swapnil Toppo | 1237143364 | Feature Engineering & Isolation Forest |
| Anoras Nancy Rajasekar | 1233650966 | Evaluation & Visualization |

## Project Overview
This project builds a hybrid unsupervised anomaly detection system for crowd 
surveillance videos. We combine optical flow-based motion modeling, a Convolutional 
Autoencoder (CAE), and Isolation Forest on handcrafted motion features to detect 
rare behavioral anomalies without manual labeling.

## Datasets
- [ShanghaiTech Campus Dataset](https://svip-lab.github.io/dataset/campus_dataset.html)
- [UCSD Pedestrian Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)

## Setup Instructions

### 1. Clone the repo
git clone https://github.com/zypher-4/crowd-anomaly-detection.git
cd crowd-anomaly-detection

### 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download datasets
See `data/README.md` for dataset download instructions.

## Project Structure
```
crowd-anomaly-detection/
├── data/
│   ├── raw/              # Original downloaded datasets
│   └── processed/        # Frames, optical flow, background subtracted
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── preprocessing/    # Frame extraction, optical flow, normalization
│   ├── models/           # Autoencoder, Isolation Forest
│   ├── evaluation/       # AUC, F1, EER metrics
│   └── visualization/    # Heatmaps, trajectory plots
├── outputs/              # Results and figures
├── requirements.txt
└── README.md
```

## Methods
1. **Optical Flow Extraction** — Farneback method via OpenCV
2. **Convolutional Autoencoder** — Trained on normal frames only
3. **Isolation Forest** — On aggregated motion features (speed, direction variance, density)

## Evaluation Metrics
- Frame-level AUC-ROC
- Precision, Recall, F1-score
- Equal Error Rate (EER)
- Detection delay
