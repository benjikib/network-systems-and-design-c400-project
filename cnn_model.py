import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('arff_file', nargs='?', default='TimeBasedFeatures-Dataset-15s-AllinOne.arff', help='path to the ARFF dataset file')
args = parser.parse_args()

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED  = 42
BATCH_SIZE   = 128
EPOCHS       = 800
LR           = 3e-4
PATIENCE     = 40
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0

torch.manual_seed(RANDOM_SEED)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data, meta = arff.loadarff(args.arff_file)
df = pd.DataFrame(data)
df = df.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

X = df.drop('class1', axis=1)
y = df['class1']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes  = len(le.classes_)
n_features = X.shape[1]
print("Classes:", le.classes_)
print(f"Features: {n_features} | Classes: {n_classes}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

def to_tensors(X_np, y_np):
    # unsqueeze to (batch, 1, features) — treat features as a 1D sequence with 1 channel
    X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_np, dtype=torch.long)
    return TensorDataset(X_t, y_t)

train_ds = to_tensors(X_train_scaled, y_train)
val_ds   = to_tensors(X_val_scaled,   y_val)
test_ds  = to_tensors(X_test_scaled,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 1D CNN — treats the feature vector as a sequence of length n_features with 1 channel.
# This is the standard way to apply conv to tabular data, but convolution assumes
# local correlations between adjacent features, which don't exist here since feature
# ordering is arbitrary. Included for comparison against the MLP.
class TrafficCNN1D(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 23, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))

model = TrafficCNN1D(n_features, n_classes).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

best_val_loss  = float('inf')
patience_count = 0
best_weights   = None
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch:3d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_weights)

def predict(loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(model(X_batch.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)

def evaluate(y_true, y_pred, split_name, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n--- {split_name} ---")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

y_val_pred = predict(val_loader)
evaluate(y_val, y_val_pred, 'CNN-1D — Validation', le.classes_)

y_test_pred = predict(test_loader)
evaluate(y_test, y_test_pred, 'CNN-1D — Test', le.classes_)

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('CNN-1D — Confusion Matrix (Test)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cnn1d_confusion_matrix.png')
plt.show()
print("Saved cnn1d_confusion_matrix.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_acc'], label='Train')
ax1.plot(history['val_acc'],   label='Val')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history['train_loss'], label='Train')
ax2.plot(history['val_loss'],   label='Val')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('cnn1d_training_curves.png')
plt.show()
print("Saved cnn1d_training_curves.png")

torch.save(model.state_dict(), 'cnn1d_model.pt')
print("Model saved as cnn1d_model.pt")
