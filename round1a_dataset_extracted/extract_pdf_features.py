#!/usr/bin/env python3
"""
(Upgraded) Extract features from each line in every PDF using PyMuPDF, for model training.
Includes advanced layout features like relative font size, whitespace, and centering.
"""

import os
import fitz  # PyMuPDF
import pandas as pd
import re

PDF_DIR = "round1a_dataset_extracted/input"
FEATURES_DIR = "round1a_dataset_extracted/features_csv"
os.makedirs(FEATURES_DIR, exist_ok=True)

def extract_features_from_pdf(pdf_path, pdf_name):
    doc = fitz.open(pdf_path)
    rows = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        
        # Calculate page-level stats for relative features
        all_font_sizes = []
        for b in blocks:
            if b['type'] == 0: # text block
                for l in b['lines']:
                    for s in l['spans']:
                        all_font_sizes.append(s['size'])
        avg_page_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 12.0 # Default if no text

        # Get text blocks for y-coord analysis
        text_blocks = sorted([b for b in blocks if b['type'] == 0 and 'lines' in b], key=lambda b: b['bbox'][1])

        for i, block in enumerate(text_blocks):
            x0, y0, x1, y1 = block['bbox']
            
            # --- New Feature: Whitespace ---
            space_before = y0 - text_blocks[i-1]['bbox'][3] if i > 0 else y0
            space_after = text_blocks[i+1]['bbox'][1] - y1 if i < len(text_blocks) - 1 else page.rect.height - y1

            for line in block['lines']:
                line_text = "".join([s['text'] for s in line['spans']]).strip()
                if not line_text:
                    continue
                
                # --- New Feature: Centering ---
                line_bbox = line['bbox']
                line_center = (line_bbox[0] + line_bbox[2]) / 2
                page_center = page.rect.width / 2
                is_centered = int(abs(line_center - page_center) < 5.0) # 5.0 is a tolerance

                # Heuristic features
                is_all_caps = int(line_text.isupper())
                is_title_case = int(line_text.istitle())
                is_numbered = int(bool(re.match(r"^\d+(\.\d+)*[\)\.]? ", line_text)))
                has_colon = int(":" in line_text)
                has_bullet = int(line_text.startswith(('-','•','*')))
                length = len(line_text)
                word_count = len(line_text.split())
                
                # Font features from spans
                font_sizes = [s['size'] for s in line['spans']]
                font_names = [s['font'] for s in line['spans']]
                is_bold = int(any("Bold" in s['font'] for s in line['spans']))
                is_italic = int(any("Italic" in s['font'] or "Oblique" in s['font'] for s in line['spans']))
                
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                most_common_font = max(set(font_names), key=font_names.count) if font_names else ""

                # --- New Feature: Relative Font Size ---
                rel_font_size = avg_font_size / avg_page_font_size if avg_page_font_size > 0 else 1.0

                rows.append({
                    "pdf": pdf_name,
                    "page": page_num,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": line_text,
                    "is_all_caps": is_all_caps,
                    "is_title_case": is_title_case,
                    "is_numbered": is_numbered,
                    "has_colon": has_colon,
                    "has_bullet": has_bullet,
                    "length": length,
                    "word_count": word_count,
                    "avg_font_size": avg_font_size,
                    "most_common_font": most_common_font,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    # New features added
                    "rel_font_size": rel_font_size,
                    "is_centered": is_centered,
                    "space_before": space_before,
                    "space_after": space_after,
                })
    return rows

def main():
    print("🔍 (Upgraded) Extracting features from all PDFs...")
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"Processing: {pdf_file}")
        try:
            rows = extract_features_from_pdf(pdf_path, pdf_file)
            if rows:
                df = pd.DataFrame(rows)
                csv_path = os.path.join(FEATURES_DIR, pdf_file.replace('.pdf', '.csv'))
                df.to_csv(csv_path, index=False)
                print(f"✅ Saved features: {csv_path}")
            else:
                print(f"⚠️ No text blocks found in {pdf_file}")
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")

    print("\n🎉 (Upgraded) Feature extraction complete!")

if __name__ == "__main__":
    main() 