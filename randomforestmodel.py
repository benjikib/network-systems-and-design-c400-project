import argparse
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('arff_file', help='path to the ARFF dataset file')
args = parser.parse_args()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load the ARFF as a dataframe
data, meta = arff.loadarff(args.arff_file)
df = pd.DataFrame(data)
df = df.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

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

# random forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# evaluate on validation set
y_val_pred_rf = rf_model.predict(X_val_scaled)

rf_accuracy = accuracy_score(y_val, y_val_pred_rf)
rf_f1       = f1_score(y_val, y_val_pred_rf, average='macro')

print(f"\nRandom Forest Validation Accuracy: {rf_accuracy:.4f}")
print(f"Random Forest Validation Macro F1: {rf_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_val, y_val_pred_rf, target_names=le.classes_))

# confusion matrix
cm_rf = confusion_matrix(y_val, y_val_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Random Forest — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as rf_confusion_matrix.png")

# feature importances
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8, 6))
plt.title('Random Forest — Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()
print("Feature importances saved as feature_importances.png")
