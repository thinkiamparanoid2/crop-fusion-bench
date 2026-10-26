import pandas as pd
import glob
import os
from collections import Counter

# 1. Path to your specific folder
base_path = r'D:\Work\Data\train'

# Searches exclusively for files ending in _translated.csv
search_pattern = os.path.join(base_path, '**', '*_translated.csv')
csv_files = glob.glob(search_pattern, recursive=True)

if not csv_files:
    print(f"No '_translated.csv' files found in {base_path}. Check if the drive is mounted or the path is correct.")
else:
    all_unique_columns = set()
    column_presence = Counter()
    total_samples = len(csv_files)

    print(f"🔍 Auditing {total_samples} multimodal samples (STRICTLY '_translated.csv' files)...")

    for f in csv_files:
        try:
            # 🚀 CHANGED LINE: Fixed the encoding to utf-8-sig based on your debug results
            df = pd.read_csv(f, low_memory=False, encoding='utf-8-sig') 
            all_unique_columns.update(df.columns)
            
            for col in df.columns:
                # A column counts as 'present' only if it has at least one non-null value
                if not df[col].isnull().all():
                    column_presence[col] += 1
        except Exception as e:
            # You should see very few, if any, errors now!
            print(f"Error reading {f}: {e}")

    # 2. Results Table
    print("\n" + "="*70)
    print(f"{'CLEANED SENSOR / COLUMN NAME':<35} | {'PRESENCE %':<12} | {'SAMPLES'}")
    print("-" * 70)
    
    # Sort columns by most frequent to least frequent
    for col, count in column_presence.most_common():
        percentage = (count / total_samples) * 100
        print(f"{col:<35} | {percentage:>10.2f}% | {count}/{total_samples}")

    # 3. Thesis Strategy Recommendation
    core_set = [col for col, count in column_presence.items() if (count/total_samples) > 0.8]
    
    print("\n" + "="*70)
    print("📋 RESEARCH RECOMMENDATION")
    print(f"To ensure your VQC (Quantum Circuit) has a stable input size,")
    print(f"use these columns as your FIXED feature set:")
    print(core_set)
    print("="*70)