import os
import pandas as pd
import glob

# 1. Define the Root Path
base_path = r'D:\Work\Data\train'

# Look for the files you just translated
search_pattern = os.path.join(base_path, '**', '*_translated.csv')
csv_files = glob.glob(search_pattern, recursive=True)

# 2. The exact columns you want to keep for your model
columns_to_keep = [
    'timestamp', 
    'int_temp_1_avg', 'int_temp_1_max', 'int_temp_1_min', 
    'int_hum_1_avg', 'int_hum_1_max', 'int_hum_1_min', 
    'int_dew_point_avg', 'int_dew_point_max', 'int_dew_point_min'
]

print(f"Starting column filtering for {len(csv_files)} files...")

for i, file_path in enumerate(csv_files):
    try:
        # Read the translated file
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Filter the DataFrame to keep ONLY the specified columns
        df_filtered = df[columns_to_keep]
        
        # Define new filename (e.g., 10017_translated.csv -> 10017_final.csv)
        new_file_path = file_path.replace('_translated.csv', '_final.csv')
        
        # Save the filtered file (index=False prevents pandas from adding row numbers)
        df_filtered.to_csv(new_file_path, index=False, encoding='utf-8')
        
        # Progress tracker
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i+1}/{len(csv_files)} files filtered.")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("✅ All files have been filtered and saved as '_final.csv'.")