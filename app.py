import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="Anomaly Detection App", layout="wide")
st.title("🔍 Anomaly Detection for Data Leakage Prevention")

# Upload data
uploaded_file = st.file_uploader("Upload your dataset (KDD format CSV)", type=["csv", "txt"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)

    st.write("### Raw Data Sample")
    st.write(df.head())

    # Add column names (based on KDD)
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
    df.columns = columns
    df.drop(["difficulty"], axis=1, inplace=True)

    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Feature Engineering
    df['feature_sum'] = df['src_bytes'] + df['dst_bytes'] + df['count']
    df['feature_product'] = df['src_bytes'] * df['dst_bytes']
    df['feature_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1e-6)
    df['feature_square'] = df['count'] ** 2
    df['feature_log'] = np.log(df['srv_count'] + 1e-6)

    df.drop(["src_bytes", "dst_bytes", "count", "srv_count"], axis=1, inplace=True)

    X = df.drop(columns=["label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb_model.fit(X_train, y_train)
    y_xgb = xgb_model.predict(X_test)

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
    mlp.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
    y_mlp = (mlp.predict(X_test) > 0.5).astype(int).flatten()

    # Ensemble
    y_ensemble = (y_rf * 0.4) + (y_xgb * 0.3) + (y_mlp * 0.3)
    y_ensemble = np.where(y_ensemble >= 0.5, 1, 0)

    # Metrics
    acc = accuracy_score(y_test, y_ensemble)
    st.write(f"### ✅ Ensemble Accuracy: `{acc:.4f}`")

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_ensemble))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_ensemble)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    st.pyplot(fig)

    # ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_ensemble)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
