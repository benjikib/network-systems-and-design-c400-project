import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('arff_file', nargs='?', default='TimeBasedFeatures-Dataset-120s.arff', help='path to the ARFF dataset file')
args = parser.parse_args()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load the ARFF as a dataframe
data, meta = arff.loadarff(args.arff_file)
df = pd.DataFrame(data)
df = df.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

# drop underperforming classes with too few samples to classify reliably
df = df[~df['class1'].isin(['STREAMING', 'VPN-CHAT'])]

# separate the features and the label
X = df.drop('class1', axis=1)
y = df['class1']

# change string labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# split the data
# split off 20% for temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
# split temp evenly into val and test (10% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# scale features
# fit on training data, then apply to val and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# baseline model - logistic regression
baseline = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
baseline.fit(X_train_scaled, y_train)

def evaluate(y_true, y_pred, split_name, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n--- {split_name} ---")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# evaluate on validation set (used for model selection)
y_val_pred = baseline.predict(X_val_scaled)
evaluate(y_val, y_val_pred, 'Baseline Logistic Regression — Validation', le.classes_)

# evaluate on held-out test set (final reported number)
y_test_pred = baseline.predict(X_test_scaled)
evaluate(y_test, y_test_pred, 'Baseline Logistic Regression — Test', le.classes_)

# confusion matrix (test set)
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Baseline Logistic Regression — Confusion Matrix (Test)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as baseline_confusion_matrix.png")
