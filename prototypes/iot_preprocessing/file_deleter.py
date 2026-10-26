import os
import glob


base_path = r'D:\Work\Data\train'

patterns_to_delete = ['*_translated.csv','*_cleaned.csv']     # Example: *_cleaned.csv


ACTUALLY_DELETE = True

total_found = 0
total_deleted = 0

print(f"Starting cleanup search in: {base_path}\n")

for pattern in patterns_to_delete:
    # Build the search path to look inside all subfolders
    search_path = os.path.join(base_path, '**', pattern)
    matched_files = glob.glob(search_path, recursive=True)
    
    if not matched_files:
        print(f"🔹 No files found matching pattern: '{pattern}'")
        continue
        
    print(f"🔹 Found {len(matched_files)} files matching '{pattern}'.")
    total_found += len(matched_files)
    
    # Loop through and handle the files
    for file_path in matched_files:
        if ACTUALLY_DELETE:
            try:
                os.remove(file_path)
                total_deleted += 1
            except Exception as e:
                print(f"  ❌ Error deleting {file_path}: {e}")

print("\n" + "="*50)
if not ACTUALLY_DELETE:
    print("⚠️ DRY RUN COMPLETE ⚠️")
    print(f"The script found {total_found} files that match your patterns.")
    print("NO FILES WERE DELETED.")
    print("If you are ready to permanently delete them, change:")
    print("ACTUALLY_DELETE = True")
    print("and run the script again.")
else:
    print("🗑️ DELETION COMPLETE 🗑️")
    print(f"Successfully deleted {total_deleted} out of {total_found} files.")
print("="*50)