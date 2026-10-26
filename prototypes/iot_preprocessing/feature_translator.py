import os
import pandas as pd
import glob

# 1. Define the Root Path
base_path = r'D:\Work\Data\train'
search_pattern = os.path.join(base_path, '**', '*_cleaned.csv')
csv_files = glob.glob(search_pattern, recursive=True)

# 2. Build the Translation Dictionary
column_map = {
    '측정시각': 'timestamp',
    '내부 이슬점 평균': 'int_dew_point_avg', '내부 이슬점 최고': 'int_dew_point_max', '내부 이슬점 최저': 'int_dew_point_min',
    '내부 CO2 평균': 'int_co2_avg', '내부 CO2 최고': 'int_co2_max', '내부 CO2 최저': 'int_co2_min',
    '외부 풍속 평균': 'ext_wind_speed_avg', '외부 풍속 최고': 'ext_wind_speed_max', '외부 풍속 최저': 'ext_wind_speed_min',
    '외부 빗물 시간': 'ext_rain_time', '외부 누적일사 평균': 'ext_cum_solar_radiation_avg',
    '양액 급액 누적': 'nutrient_supply_cum', '양액 배액 누적': 'nutrient_drainage_cum',
    '배지 중량 평균': 'substrate_weight_avg', '배지 중량 최고': 'substrate_weight_max', '배지 중량 최저': 'substrate_weight_min',
    '양액 온도 평균': 'nutrient_temp_avg', '양액 온도 최고': 'nutrient_temp_max', '양액 온도 최저': 'nutrient_temp_min',
    '외부 풍향 수치': 'ext_wind_dir_val', '외부 풍향': 'ext_wind_dir_name'
}

# Add dynamic mapping for numbered sensors (1-4)
for i in range(1, 5):
    column_map.update({
        f'내부 온도 {i} 평균': f'int_temp_{i}_avg', f'내부 온도 {i} 최고': f'int_temp_{i}_max', f'내부 온도 {i} 최저': f'int_temp_{i}_min',
        f'내부 습도 {i} 평균': f'int_hum_{i}_avg', f'내부 습도 {i} 최고': f'int_hum_{i}_max', f'내부 습도 {i} 최저': f'int_hum_{i}_min',
        f'내부 EC {i} 평균': f'int_ec_{i}_avg', f'내부 EC {i} 최고': f'int_ec_{i}_max', f'내부 EC {i} 최저': f'int_ec_{i}_min',
        f'내부 PH {i} 평균': f'int_ph_{i}_avg', f'내부 PH {i} 최고': f'int_ph_{i}_max', f'내부 PH {i} 최저': f'int_ph_{i}_min',
    })

# 3. Execution Loop
print(f"Starting translation for {len(csv_files)} files...")

for i, file_path in enumerate(csv_files):
    try:
        # Read the cleaned file
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Rename the columns
        df = df.rename(columns=column_map)
        
        # Define new filename (e.g., 10017_cleaned.csv -> 10017_translated.csv)
        new_file_path = file_path.replace('_cleaned.csv', '_translated.csv')
        
        # Save the file (index=False prevents adding an extra column)
        df.to_csv(new_file_path, index=False, encoding='utf-8')
        
        # Progress tracker
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i+1}/{len(csv_files)} files translated.")
            
    except Exception as e:
        print(f"Error translating {file_path}: {e}")

print("✅ All files have been translated and saved as '_translated.csv'.")