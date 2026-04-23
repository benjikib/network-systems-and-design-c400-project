import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('arff_file', help='path to the ARFF dataset file')
args = parser.parse_args()
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 42
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-3
PATIENCE    = 10

torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# load the ARFF as a dataframe
data, meta = arff.loadarff(args.arff_file)
df = pd.DataFrame(data)
df = df.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

# separate the features and the label
X = df.drop('class1', axis=1)
y = df['class1']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes  = len(le.classes_)
n_features = X.shape[1]
print("Classes:", le.classes_)
print(f"Features: {n_features} | Classes: {n_classes}")

# split the data
# split off 20% for temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# scale features
# fit on training data, then apply to val and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# build pytorch datasets
# CNN expects (batch, channels, length) - reshape to (N, 1, n_features)
def to_tensors(X_np, y_np):
    X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_np, dtype=torch.long)
    return TensorDataset(X_t, y_t)

train_ds = to_tensors(X_train_scaled, y_train)
val_ds   = to_tensors(X_val_scaled,   y_val)
test_ds  = to_tensors(X_test_scaled,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# define the 1D CNN
class TrafficCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = TrafficCNN(n_features, n_classes).to(device)
print(model)

# training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5, min_lr=1e-6
)

# train loop with early stopping
best_val_loss   = float('inf')
patience_count  = 0
best_weights    = None
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
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch:3d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_weights)

# evaluate on validation set
def predict(loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(model(X_batch.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)

y_val_pred = predict(val_loader)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1       = f1_score(y_val, y_val_pred, average='macro')

print(f"\nCNN Validation Accuracy: {val_accuracy:.4f}")
print(f"CNN Validation Macro F1: {val_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('CNN — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png')
plt.show()
print("Saved cnn_confusion_matrix.png")

# training curves
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
plt.savefig('cnn_training_curves.png')
plt.show()
print("Saved cnn_training_curves.png")

# save model weights
torch.save(model.state_dict(), 'cnn_model.pt')
print("Model saved as cnn_model.pt")
