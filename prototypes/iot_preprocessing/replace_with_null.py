import pandas as pd
import numpy as np
from pathlib import Path

def replace_dashes_with_null(root_folder_path):
    root_dir = Path(root_folder_path)
    
    print("Scanning for original CSV files...")
    
    # Search for all .csv files, but safely ignore any that have already been cleaned
    csv_files = [f for f in root_dir.rglob("*.csv") if not f.name.endswith("_cleaned.csv")]
    total_files = len(csv_files)
    
    if total_files == 0:
        print("No original CSV files found to process.")
        return
        
    print(f"Found {total_files} CSV files. Starting replacement...")
    
    for index, file_path in enumerate(csv_files, start=1):
        try:
            # 1. Read the current CSV file
            df = pd.read_csv(file_path)
            
            # 2. Replace all exact "-" values with true nulls (NaN) across the entire file
            df.replace("-", np.nan, inplace=True)
            
            # 3. Create the new filename (e.g., "10063.csv" -> "10063_cleaned.csv")
            new_filename = f"{file_path.stem}_cleaned.csv"
            
            # 4. Define the exact same folder for the new file to keep it with the photo
            new_file_path = file_path.parent / new_filename
            
            # 5. Export the new CSV
            df.to_csv(new_file_path, index=False)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
        # Print a progress update every 1,000 files
        if index % 1000 == 0:
            print(f"Processed {index} / {total_files} files...")

    print("Success! All dashes have been replaced with nulls and saved as new files.")

# --- RUN THE CODE ---
# Replace this path with the path to your main folder containing all the subfolders
target_directory = r"D:\Work\Data\train" 

replace_dashes_with_null(target_directory)