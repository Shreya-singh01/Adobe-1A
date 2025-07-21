#!/usr/bin-python3
"""
(Fortified) Inference script: Given a PDF, extract features, predict header levels with a fine-tuned XGBoost model, and apply a rule-based layer to catch errors and improve accuracy on real-world PDFs.
"""

import os
import sys
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'header_classifier_xgb_finetuned.model')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'header_label_encoder_finetuned.joblib')

FEATURE_COLS = [
    'page', 'x0', 'y0', 'x1', 'y1', 'is_all_caps', 'is_title_case', 'is_numbered',
    'has_colon', 'has_bullet', 'length', 'word_count', 'avg_font_size', 'is_bold', 'is_italic', 
    'rel_font_size', 'is_centered', 'space_before', 'space_after', 'most_common_font'
]

# --- Feature Extraction ---
def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    rows = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        all_font_sizes = []
        for b in blocks:
            if b['type'] == 0:
                for l in b['lines']:
                    for s in l['spans']:
                        all_font_sizes.append(s['size'])
        avg_page_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 12.0
        text_blocks = sorted([b for b in blocks if b['type'] == 0 and 'lines' in b], key=lambda b: b['bbox'][1])
        for i, block in enumerate(text_blocks):
            x0, y0, x1, y1 = block['bbox']
            space_before = y0 - text_blocks[i-1]['bbox'][3] if i > 0 else y0
            space_after = text_blocks[i+1]['bbox'][1] - y1 if i < len(text_blocks) - 1 else page.rect.height - y1
            for line in block['lines']:
                line_text = "".join([s['text'] for s in line['spans']]).strip()
                if not line_text:
                    continue
                line_bbox = line['bbox']
                line_center = (line_bbox[0] + line_bbox[2]) / 2
                page_center = page.rect.width / 2
                is_centered = int(abs(line_center - page_center) < 5.0)
                is_all_caps = int(line_text.isupper())
                is_title_case = int(line_text.istitle())
                is_numbered = int(bool(re.match(r"^\d+(\.\d+)*[\)\.]? ", line_text)))
                has_colon = int(":" in line_text)
                has_bullet = int(line_text.startswith(('-','•','*')))
                length = len(line_text)
                word_count = len(line_text.split())
                font_sizes = [s['size'] for s in line['spans']]
                font_names = [s['font'] for s in line['spans']]
                is_bold = int(any("Bold" in s['font'] for s in line['spans']))
                is_italic = int(any("Italic" in s['font'] or "Oblique" in s['font'] for s in line['spans']))
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                most_common_font = max(set(font_names), key=font_names.count) if font_names else ""
                rel_font_size = avg_font_size / avg_page_font_size if avg_page_font_size > 0 else 1.0
                rows.append({
                    "page": page_num, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": line_text,
                    "is_all_caps": is_all_caps, "is_title_case": is_title_case, "is_numbered": is_numbered,
                    "has_colon": has_colon, "has_bullet": has_bullet, "length": length, "word_count": word_count,
                    "avg_font_size": avg_font_size, "is_bold": is_bold, "is_italic": is_italic,
                    "most_common_font": most_common_font, "rel_font_size": rel_font_size, 
                    "is_centered": is_centered, "space_before": space_before, "space_after": space_after
                })
    return pd.DataFrame(rows)

# --- Inference ---
def predict_headers(df, model, label_encoder):
    font_le = LabelEncoder()
    df['most_common_font'] = font_le.fit_transform(df['most_common_font'].astype(str))
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].values
    y_pred = model.predict(X)
    labels = label_encoder.inverse_transform(y_pred)
    df['pred_label'] = labels
    return df

# --- Rule-based Fortification Layer ---
def fortify_predictions(df):
    new_labels = df['pred_label'].copy()
    for i, row in df.iterrows():
        # Rule 1: Big Font Rule
        if row['rel_font_size'] > 1.5:
            new_labels[i] = 'H1'
        # Rule 2: Centered Text Rule
        if row['is_centered'] and row['avg_font_size'] >= 12 and new_labels[i] == 'BODY':
            new_labels[i] = 'H2'
        # Rule 3: Whitespace Rule
        if row['space_before'] > 20 and new_labels[i] == 'BODY': # 20 is a threshold
            new_labels[i] = 'H2'
    # Rule 4: De-duplication Rule
    for i in range(1, len(new_labels)):
        if new_labels[i] == 'H1' and new_labels[i-1] == 'H1':
            new_labels[i] = 'H2'
    df['fortified_label'] = new_labels
    return df

# --- JSON Output ---
def build_json_outline(df, pdf_path):
    title = os.path.splitext(os.path.basename(pdf_path))[0]
    for t in df['text']:
        if len(t) > 5 and len(t) < 100:
            title = t
            break
    outline = []
    for _, row in df.iterrows():
        if row['fortified_label'] in ['H1', 'H2', 'H3']:
            outline.append({
                'level': row['fortified_label'],
                'text': row['text'],
                'page': int(row['page'])
            })
    return {
        'title': title,
        'outline': outline
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python infer_pdf_outline.py <input.pdf> [output.json]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else pdf_path.replace('.pdf', '_fortified_outline.json')

    print(f"🔍 Extracting features from {pdf_path} ...")
    df = extract_features_from_pdf(pdf_path)
    print(f"🧠 Loading fine-tuned model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print(f"🔮 Predicting headers...")
    df_pred = predict_headers(df, model, label_encoder)
    print(f"🛡️ Fortifying predictions with rule-based layer...")
    df_fortified = fortify_predictions(df_pred)
    print(f"📝 Building JSON outline...")
    outline_json = build_json_outline(df_fortified, pdf_path)
    with open(output_json, 'w') as f:
        json.dump(outline_json, f, indent=4)
    print(f"✅ Saved fortified outline to {output_json}")

if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder
    main() 