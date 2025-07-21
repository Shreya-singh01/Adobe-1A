#!/usr/bin/env python3
"""
(Upgraded) Train an XGBoost classifier for header detection using labeled features.
Includes advanced layout features for improved accuracy.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

FEATURES_DIR = "round1a_dataset_extracted/features_csv"
MODEL_PATH = "header_classifier_xgb_finetuned.model"

# Features to use for training (now including advanced layout features)
FEATURE_COLS = [
    'page', 'x0', 'y0', 'x1', 'y1', 
    'is_all_caps', 'is_title_case', 'is_numbered',
    'has_colon', 'has_bullet', 
    'length', 'word_count', 
    'avg_font_size', 'is_bold', 'is_italic',
    # New features
    'rel_font_size', 'is_centered', 'space_before', 'space_after'
]

# Categorical features to encode
CATEGORICAL_COLS = ['most_common_font']

LABEL_COL = 'label'


def load_labeled_data():
    print("📥 Loading labeled data...")
    dfs = []
    for fname in os.listdir(FEATURES_DIR):
        if fname.endswith('_labeled.csv'):
            df = pd.read_csv(os.path.join(FEATURES_DIR, fname))
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} lines from {len(dfs)} PDFs.")
    return data

def prepare_features(data):
    # Encode categorical features
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    
    all_features = FEATURE_COLS + CATEGORICAL_COLS
    
    # Ensure all columns are present
    for col in all_features:
        if col not in data.columns:
            data[col] = 0 # Or some other default value

    X = data[all_features].values
    y = data[LABEL_COL].values
    return X, y

def main():
    data = load_labeled_data()
    X, y = prepare_features(data)
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Save label encoder for inference
    joblib.dump(label_encoder, 'header_label_encoder_finetuned.joblib')
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    # Train XGBoost classifier
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(X_train, y_train)
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report (Fine-tuned Model):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix (Fine-tuned Model):")
    print(confusion_matrix(y_test, y_pred))
    # Save model
    clf.save_model(MODEL_PATH)
    print(f"\n✅ (Fine-tuned) Model saved to {MODEL_PATH}")
    print(f"✅ (Fine-tuned) Label encoder saved to header_label_encoder_finetuned.joblib")

if __name__ == "__main__":
    main() 