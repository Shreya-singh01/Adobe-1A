#!/usr/bin/env python3
"""
Label each line in the features CSVs as H1, H2, H3, or BODY using the ground-truth JSONs.
"""

import os
import pandas as pd
import json
from difflib import SequenceMatcher

FEATURES_DIR = "round1a_dataset_extracted/features_csv"
JSON_DIR = "round1a_dataset_extracted/output"

LABELS = ["H1", "H2", "H3"]

# Helper: fuzzy match for header text

def is_header_match(line_text, header_text):
    # Normalize and compare
    line = line_text.strip().lower()
    header = header_text.strip().lower()
    # Allow for minor differences (e.g., punctuation, whitespace)
    ratio = SequenceMatcher(None, line, header).ratio()
    return ratio > 0.85  # 85%+ similarity

def label_features_for_pdf(csv_path, json_path):
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        outline = json.load(f)['outline']
    # Build lookup: (page, normalized header text) -> label
    header_lookup = {}
    for h in outline:
        key = (h['page'], h['text'].strip().lower())
        header_lookup[key] = h['level']
    # Assign labels
    labels = []
    for _, row in df.iterrows():
        found = False
        for h in outline:
            if row['page'] == h['page'] and is_header_match(row['text'], h['text']):
                labels.append(h['level'])
                found = True
                break
        if not found:
            labels.append('BODY')
    df['label'] = labels
    return df

def main():
    print("🔖 Labeling features with H1, H2, H3, BODY...")
    for csv_file in os.listdir(FEATURES_DIR):
        if not csv_file.endswith('.csv'):
            continue
        base = csv_file.replace('.csv', '')
        json_file = base + '.json'
        csv_path = os.path.join(FEATURES_DIR, csv_file)
        json_path = os.path.join(JSON_DIR, json_file)
        if not os.path.exists(json_path):
            print(f"❌ JSON not found for {csv_file}")
            continue
        print(f"Labeling: {csv_file}")
        df_labeled = label_features_for_pdf(csv_path, json_path)
        labeled_csv_path = os.path.join(FEATURES_DIR, base + '_labeled.csv')
        df_labeled.to_csv(labeled_csv_path, index=False)
        print(f"✅ Saved: {labeled_csv_path}")
    print("\n🎉 All features labeled!")

if __name__ == "__main__":
    main() 