import pandas as pd
import glob
import os

# 1. Path to your filtered files
base_path = r'D:\Work\Data\test'
search_pattern = os.path.join(base_path, '**', '*_final.csv')
csv_files = glob.glob(search_pattern, recursive=True)

columns_to_scale = [
    'int_temp_1_avg', 'int_temp_1_max', 'int_temp_1_min', 
    'int_hum_1_avg', 'int_hum_1_max', 'int_hum_1_min', 
    'int_dew_point_avg', 'int_dew_point_max', 'int_dew_point_min'
]

print(f"Calculating global statistics across {len(csv_files)} files...")

# Initialize global min and max dictionaries
global_min = {col: float('inf') for col in columns_to_scale}
global_max = {col: float('-inf') for col in columns_to_scale}

for i, file_path in enumerate(csv_files):
    try:
        df = pd.read_csv(file_path)
        
        for col in columns_to_scale:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min < global_min[col]: global_min[col] = col_min
            if col_max > global_max[col]: global_max[col] = col_max
            
    except Exception as e:
        pass

print("\n" + "="*50)
print("🌍 GLOBAL MIN / MAX FOR NORMALIZATION (MinMax Scaler)")
print("="*50)
for col in columns_to_scale:
    print(f"'{col}': Min = {global_min[col]:.2f}, Max = {global_max[col]:.2f}")
print("="*50)