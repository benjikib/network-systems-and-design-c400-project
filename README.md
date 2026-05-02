# Network Systems And Design C400 Project
> Benji Kiblinger

## Comparison of Machine Learning Models for Classifying Encrypted and Non-Encrypted Network Traffic

This project compares five machine learning models for classifying network traffic across seven application categories (browsing, chat, file transfer, email, P2P, streaming, VoIP) using the ISCX VPN-nonVPN dataset. The models are evaluated on both VPN-encrypted and non-encrypted traffic using flow-level statistical features, without any payload inspection.

**Models compared:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- 1D Convolutional Neural Network (CNN)
- Multi-Layer Perceptron (MLP)

**Best result:** XGBoost — 90.4% accuracy, 0.87 macro F1-score

### Setup

Create a Python virtual environment and install the required dependencies:

```bash
python3 -m venv capstone_env
source capstone_env/bin/activate
pip install -r requirements.txt
```

### Running the Models

Each model is a standalone script. All default to the 15-second AllinOne dataset:

```bash
python baseline.py                                          # logistic regression
python randomforestmodel.py                                 # random forest
python xgboost_model.py                                     # xgboost
python cnn_model.py                                         # 1D CNN
python mlp_model.py                                         # MLP
```

To use a different ARFF file, pass it as an argument:

```bash
python xgboost_model.py TimeBasedFeatures-Dataset-120s-AllinOne.arff
```

Figures are saved to `figures/` and model weights to `models/`.

### Dataset

The ISCX VPN-nonVPN dataset, available from the University of New Brunswick, contains labeled bidirectional network flows extracted using ISCXFlowMeter across 15s, 30s, 60s, and 120s time windows.

([Dataset](https://www.unb.ca/cic/datasets/vpn.html))
