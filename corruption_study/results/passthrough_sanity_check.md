# Passthrough Sanity Check

Passthrough materializes resized PNGs without corruption and verifies the image path wiring.

Tolerance: `0.005` absolute delta for accuracy and Macro-F1.

| model   |   clean_accuracy |   passthrough_accuracy |   accuracy_delta |   clean_macro_f1 |   passthrough_macro_f1 |   macro_f1_delta | status   |
|:--------|-----------------:|-----------------------:|-----------------:|-----------------:|-----------------------:|-----------------:|:---------|
| e0      |          0.98214 |               0.982487 |      0.000346801 |         0.963896 |               0.964976 |        0.0010794 | PASS     |
