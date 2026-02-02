import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

def preprocess_data(df, encoder=None, is_train=True):
    """Applies common preprocessing steps to both train and test data."""
    df.columns = columns
    df.drop(["difficulty"], axis=1, inplace=True)


    categorical_cols = ["protocol_type", "service", "flag"]
    if is_train:
        encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = encoder.fit_transform(df[col])
    else:
        for col in categorical_cols:
            for i, label in enumerate(df[col]):
                if label not in encoder.classes_:
                    df.loc[df[col] == label, col] = encoder.classes_[0] 
            df[col] = encoder.transform(df[col])

    # Encode label: 0 = normal, 1 = attack
    # KDD labels are 'normal.' or 'attack_name'
    df["label"] = df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)

    # Feature engineering 
    numeric_features = ["src_bytes", "dst_bytes", "count", "srv_count"]
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['feature_sum'] = df['src_bytes'] + df['dst_bytes'] + df['count']
    df['feature_product'] = df['src_bytes'] * df['dst_bytes']
    df['feature_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1e-6)
    df['feature_square'] = df['count'] ** 2
    df['feature_log'] = np.log(df['srv_count'].clip(lower=1) + 1e-6)

    df.drop(numeric_features, axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute NaNs using mean from the current data 
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df.drop(columns=["label"])
    y = df["label"]

    return X, y, encoder

# --- 1. Load Data ---
try:
    # Load Training Data
    df_train = pd.read_csv("KDDTrain+.txt.zip", header=None)
    # Load Testing Data
    df_test = pd.read_csv("KDDTest+.txt.zip", header=None)
except FileNotFoundError:
    print("Error: One or both KDD files not found. Please upload 'KDDTrain+.txt.zip' and 'KDDTest+.txt.zip'.")
    exit()

# --- 2. Preprocess Data ---
# Preprocess Training Data (Fit encoders)
X_train_orig, y_train, encoder = preprocess_data(df_train, is_train=True)

# Preprocess Testing Data (Use fitted encoders, DO NOT fit)
X_test_orig, y_test, _ = preprocess_data(df_test, encoder=encoder, is_train=False)

print("\nClass distribution in training set:")
print(y_train.value_counts())
print("\nClass distribution in testing set:")
print(y_test.value_counts())


# --- 3. Standardization (Fit on Train, Transform Test) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_orig)
X_test_scaled = scaler.transform(X_test_orig)


# --- 4. PCA (Fit on Train, Transform Test) ---
n_components_pca = min(10, X_train_scaled.shape[1])
pca = PCA(n_components=n_components_pca)

X_train = pca.fit_transform(X_train_scaled)
X_test = pca.transform(X_test_scaled)


# --- 5. Train Individual Models (and get probabilities/scores) ---
print("\n--- Training Individual Models ---")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_rf_proba = rf.predict_proba(X_test)[:, 1]
y_rf = (y_rf_proba > 0.5).astype(int)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
y_xgb = (y_xgb_proba > 0.5).astype(int)

# Autoencoder (Anomaly Detection)
X_train_normal = X_train[y_train == 0]
autoencoder = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(16, activation="relu"),
    Dense(32, activation="relu"),
    Dense(X_train.shape[1], activation="sigmoid")
])
autoencoder.compile(optimizer="adam", loss="mse")
print("Training Autoencoder...")
autoencoder.fit(X_train_normal, X_train_normal, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

X_test_pred = autoencoder.predict(X_test, verbose=0)
reconstruction_error = np.mean(np.square(X_test_pred - X_test), axis=1)

# Normalize reconstruction error for ensemble score (0 to 1)
y_auto_proba = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min() + 1e-6)

# Set threshold for binary prediction accuracy reporting
threshold = np.percentile(reconstruction_error, 100 - (y_train.sum() / len(y_train) * 100 * 1.5))
y_auto = np.where(reconstruction_error > threshold, 1, 0)

# MLP
mlp = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])
mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("Training MLP...")
mlp.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
y_mlp_proba = mlp.predict(X_test, verbose=0).flatten()
y_mlp = (y_mlp_proba > 0.5).astype(int)


# --- 6. Automated Ensemble Weights (Trained on Validation Split of Training Data) ---
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Predictions/Scores on validation set
y_rf_val_proba = rf.predict_proba(X_val)[:, 1]
y_xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
reconstruction_error_val = np.mean(np.square(autoencoder.predict(X_val, verbose=0) - X_val), axis=1)
y_auto_val_proba = (reconstruction_error_val - reconstruction_error_val.min()) / (reconstruction_error_val.max() - reconstruction_error_val.min() + 1e-6)
y_mlp_val_proba = mlp.predict(X_val, verbose=0).flatten()

y_val_probas = [y_rf_val_proba, y_xgb_val_proba, y_auto_val_proba, y_mlp_val_proba]


def find_best_weights_normalized(y_probas, y_true):
    best_acc = 0
    best_w = None
    steps = np.arange(0.0, 1.1, 0.1)

    print("\nSearching for best ensemble weights...")
    for w_rf in steps:
        for w_xgb in steps:
            for w_auto in steps:
                w_mlp = 1.0 - w_rf - w_xgb - w_auto

                if w_mlp < 0 or w_mlp > 1.0:
                    continue

                current_weights = (w_rf, w_xgb, w_auto, w_mlp)

                y_comb_proba = (y_probas[0]*w_rf + y_probas[1]*w_xgb +
                                y_probas[2]*w_auto + y_probas[3]*w_mlp)
                y_comb = np.where(y_comb_proba >= 0.5, 1, 0)
                acc = accuracy_score(y_true, y_comb)

                if acc > best_acc:
                    best_acc = acc
                    best_w = current_weights
    return best_w, best_acc

best_weights, best_acc = find_best_weights_normalized(y_val_probas, y_val)
print(f"\nBest ensemble weights (RF, XGB, Auto, MLP): {best_weights}")


# Apply best weights on Test Set using PROBABILITIES
y_test_probas = [y_rf_proba, y_xgb_proba, y_auto_proba, y_mlp_proba]

y_ensemble_proba = (best_weights[0]*y_test_probas[0] + best_weights[1]*y_test_probas[1] +
                    best_weights[2]*y_test_probas[2] + best_weights[3]*y_test_probas[3])
y_ensemble_auto = np.where(y_ensemble_proba >= 0.5, 1, 0)


# --- 7. Evaluation ---
print("\n" + "="*50)
print("FINAL TEST PERFORMANCE SUMMARY (Trained on KDDTrain+, Tested on KDDTest+)")
print("="*50)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_rf):.4f}")
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_xgb):.4f}")
print(f"Autoencoder Accuracy: {accuracy_score(y_test, y_auto):.4f}")
print(f"MLP Accuracy: {accuracy_score(y_test, y_mlp):.4f}")
print("-" * 50)
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_ensemble_auto):.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_ensemble_auto))

# Confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_ensemble_auto), annot=True, fmt="d", cmap="Blues",
             xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ensemble")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_ensemble_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ensemble")
plt.legend()
plt.show()
