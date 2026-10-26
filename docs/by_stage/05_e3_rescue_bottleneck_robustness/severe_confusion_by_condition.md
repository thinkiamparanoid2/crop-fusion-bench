# Severe Confusion by IoT Condition

| Model | Condition | True severe | Pred healthy | Pred early | Pred mid | Pred severe | Severe recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2_matched_8 | clean | 56 | 0 | 5 | 6 | 45 | 0.803571 |
| e3_8_vqc | clean | 56 | 0 | 9 | 2 | 45 | 0.803571 |
| e2_matched_8 | gaussian_noise_5 | 56 | 0 | 5 | 6 | 45 | 0.803571 |
| e3_8_vqc | gaussian_noise_5 | 56 | 0 | 9 | 2 | 45 | 0.803571 |
| e2_matched_8 | gaussian_noise_10 | 56 | 0 | 5 | 6 | 45 | 0.803571 |
| e3_8_vqc | gaussian_noise_10 | 56 | 0 | 9 | 2 | 45 | 0.803571 |
| e2_matched_8 | gaussian_noise_15 | 56 | 0 | 5 | 5 | 46 | 0.821429 |
| e3_8_vqc | gaussian_noise_15 | 56 | 0 | 9 | 2 | 45 | 0.803571 |
| e2_matched_8 | missing_10 | 56 | 0 | 6 | 5 | 45 | 0.803571 |
| e3_8_vqc | missing_10 | 56 | 0 | 8 | 3 | 45 | 0.803571 |
| e2_matched_8 | missing_20 | 56 | 0 | 6 | 4 | 46 | 0.821429 |
| e3_8_vqc | missing_20 | 56 | 0 | 9 | 2 | 45 | 0.803571 |
| e2_matched_8 | shuffled_iot_20 | 56 | 0 | 6 | 6 | 44 | 0.785714 |
| e3_8_vqc | shuffled_iot_20 | 56 | 1 | 14 | 3 | 38 | 0.678571 |
| e2_matched_8 | contradictory_iot_10 | 56 | 1 | 4 | 6 | 45 | 0.803571 |
| e3_8_vqc | contradictory_iot_10 | 56 | 1 | 9 | 2 | 44 | 0.785714 |
| e2_matched_8 | combined_noise10_missing10_contradictory10 | 56 | 0 | 5 | 5 | 46 | 0.821429 |
| e3_8_vqc | combined_noise10_missing10_contradictory10 | 56 | 0 | 11 | 3 | 42 | 0.750000 |
