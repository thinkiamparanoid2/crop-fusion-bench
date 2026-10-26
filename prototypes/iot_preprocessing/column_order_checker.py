import pandas as pd
from pathlib import Path

def check_cleaned_column_consistency(root_folder_path):
    root_dir = Path(root_folder_path)
    
    
    csv_files = list(root_dir.rglob("*_final.csv"))
    
    if not csv_files:
        print("No cleaned CSV files found! Double-check your export step.")
        return

    # Use the first cleaned file as the "Master Template"
    first_file = csv_files[0]
    master_columns = list(pd.read_csv(first_file, nrows=0).columns)
    
    print(f"Master Reference (Cleaned): {first_file.name}")
    print(f"Master Columns: {master_columns}\n")
    
    mismatched_files = []

    print(f"Checking {len(csv_files)} cleaned files for consistency...")

    for file_path in csv_files:
        try:
            # Efficiently read only the header
            current_columns = list(pd.read_csv(file_path, nrows=0).columns)
            
            if current_columns != master_columns:
                mismatched_files.append((file_path.name, current_columns))
        
        except Exception as e:
            print(f"Could not read {file_path.name}: {e}")

    # --- FINAL REPORT ---
    if not mismatched_files:
        print("✅ SUCCESS: All cleaned files have identical columns and order.")
    else:
        print(f"❌ Found {len(mismatched_files)} files that do not match the master template.")
        print("-" * 50)
        for name, cols in mismatched_files[:10]: # Showing first 10 errors
            print(f"File: {name}")
            print(f"Columns: {cols}\n")
        
        if len(mismatched_files) > 10:
            print(f"...and {len(mismatched_files) - 10} more files with mismatches.")

# Use the same main folder address as before
target_directory = r"D:\Work\data\test" 
check_cleaned_column_consistency(target_directory)