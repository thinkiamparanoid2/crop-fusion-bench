# Severity-0 Sanity Check

Severity 0 evaluates the original clean test inputs and should reproduce the audited clean metrics.

Tolerance: `0.005` absolute delta for accuracy and Macro-F1.

| model        |   accuracy |   expected_accuracy |   accuracy_delta |   macro_f1 |   expected_macro_f1 |   macro_f1_delta | status   |
|:-------------|-----------:|--------------------:|-----------------:|-----------:|--------------------:|-----------------:|:---------|
| e0           |   0.98214  |            0.98214  |                0 |   0.963896 |            0.963896 |                0 | PASS     |
| e1           |   0.943645 |            0.943645 |                0 |   0.88337  |            0.88337  |                0 | PASS     |
| e2_canonical |   0.987342 |            0.987342 |                0 |   0.967926 |            0.967926 |                0 | PASS     |
| e2_8         |   0.987515 |            0.987515 |                0 |   0.973356 |            0.973356 |                0 | PASS     |
