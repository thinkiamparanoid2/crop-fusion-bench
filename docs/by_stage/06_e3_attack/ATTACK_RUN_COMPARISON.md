# E3 Attack Run Comparison

E2-matched-8 test Macro F1 bar: 0.973356.

| Run | Model | Checkpoint | Seed | Post params | Alpha mode | Alpha | Accuracy | Macro F1 | Macro recall | Early | Mid | Severe | ECE | Beats E2 bar |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| e2_matched_8_residual_bounded_diag_s42 | e2_matched residual | best_macro | 42 | 65 | bounded <= 1 | 0.711385 | 0.987689 | 0.973469 | 0.971665 | 0.985682 | 0.922727 | 0.821429 | 0.060777 | yes |
| e2_matched_8_residual_bounded_diag_s42 | e2_matched residual | best_mid | 42 | 65 | bounded <= 1 | 0.711385 | 0.987689 | 0.973469 | 0.971665 | 0.985682 | 0.922727 | 0.821429 | 0.060777 | yes |
| e2_matched_8_pair_s42 | e2_matched | best_macro | 42 | 64 |  | nan | 0.987862 | 0.973235 | 0.970534 | 0.980407 | 0.934091 | 0.803571 | 0.149690 | no |
| e2_matched_8_residual_bounded_pair_diag_s42 | e2_matched residual | best_macro | 42 | 65 | bounded <= 1 | 0.709568 | 0.987342 | 0.972314 | 0.970814 | 0.979653 | 0.936364 | 0.803571 | 0.061566 | no |
| e2_matched_8_residual_diag_s42 | e2_matched residual | best_macro | 42 | 65 | free | 0.807556 | 0.987342 | 0.972224 | 0.970371 | 0.985682 | 0.920455 | 0.803571 | 0.073810 | no |
| e2_matched_8_residual_diag_s42 | e2_matched residual | best_mid | 42 | 65 | free | 0.807556 | 0.987342 | 0.972224 | 0.970371 | 0.985682 | 0.920455 | 0.803571 | 0.073810 | no |
| e2_matched_8_pair_s42 | e2_matched | best_mid | 42 | 64 |  | nan | 0.986475 | 0.971065 | 0.969237 | 0.966843 | 0.965909 | 0.785714 | 0.148803 | no |
| e3_8_warm_residual_bounded_pair_diag_s42 | e3_vqc residual | best_macro | 42 | 65 | bounded <= 1 | 0.999536 | 0.985261 | 0.968655 | 0.968186 | 0.975132 | 0.936364 | 0.803571 | 0.108665 | no |
| e3_8_warm_residual_bounded_pair_a15_l1_diag_s42 | e3_vqc residual | best_macro | 42 | 65 | bounded <= 1.5 | 1.499606 | 0.985434 | 0.967377 | 0.966447 | 0.973625 | 0.929545 | 0.785714 | 0.058427 | no |
| e3_8_warm_residual_bounded_diag_s42 | e3_vqc residual | best_macro | 42 | 65 | bounded <= 1 | 0.999894 | 0.984047 | 0.966532 | 0.966741 | 0.978146 | 0.929545 | 0.785714 | 0.156215 | no |
| e3_8_warm_residual_bounded_pair_l3_diag_s42 | e3_vqc residual | best_macro | 42 | 65 | bounded <= 1 | 0.999712 | 0.984914 | 0.965295 | 0.962655 | 0.972871 | 0.956818 | 0.678571 | 0.075959 | no |
| e3_8_warm_residual_diag_s42 | e3_vqc residual | best_macro | 42 | 65 | free | 13.611519 | 0.981620 | 0.961233 | 0.958809 | 0.974378 | 0.913636 | 0.732143 | 0.009437 | no |
| e3_8_warm_pair_s42 | e3_vqc | best_macro | 42 | 64 |  | nan | 0.982140 | 0.957459 | 0.961289 | 0.969103 | 0.938636 | 0.732143 | 0.140382 | no |
| e3_8_warm_residual_diag_s42 | e3_vqc residual | best_mid | 42 | 65 | free | 13.416590 | 0.981793 | 0.957425 | 0.962314 | 0.978146 | 0.915909 | 0.803571 | 0.011787 | no |
| e3_8_warm_s42 | e3_8_vqc_warmstart | best_macro | 42 | 64 |  | nan | 0.981446 | 0.956587 | 0.959699 | 0.976639 | 0.902273 | 0.821429 | 0.229923 | no |
| e3_8_warm_s42 | e3_8_vqc_warmstart | best_mid | 42 | 64 |  | nan | 0.982140 | 0.953520 | 0.957938 | 0.975885 | 0.925000 | 0.732143 | 0.216430 | no |
| e3_8_warm_residual_bounded_diag_s42 | e3_vqc residual | best_mid | 42 | 65 | bounded <= 1 | 0.999514 | 0.971389 | 0.947028 | 0.954862 | 0.964582 | 0.918182 | 0.785714 | 0.342218 | no |
| e2_matched_8_residual_bounded_pair_diag_s42 | e2_matched residual | best_mid | 42 | 65 | bounded <= 1 | 0.668956 | 0.980579 | 0.879601 | 0.889411 | 0.960814 | 0.947727 | 0.553571 | 0.139041 | no |
| e3_8_warm_pair_s42 | e3_vqc | best_mid | 42 | 64 |  | nan | 0.931333 | 0.861886 | 0.892507 | 0.761869 | 0.970455 | 0.678571 | 0.310006 | no |
| e3_8_warm_residual_bounded_pair_l3_diag_s42 | e3_vqc residual | best_mid | 42 | 65 | bounded <= 1 | 0.998714 | 0.890411 | 0.745983 | 0.805148 | 0.772419 | 0.986364 | 0.482143 | 0.324857 | no |
| e3_8_warm_residual_bounded_pair_a15_l1_diag_s42 | e3_vqc residual | best_mid | 42 | 65 | bounded <= 1.5 | 1.493877 | 0.749090 | 0.681905 | 0.811051 | 0.791258 | 0.977273 | 0.464286 | 0.217731 | no |
| e3_8_warm_residual_bounded_pair_diag_s42 | e3_vqc residual | best_mid | 42 | 65 | bounded <= 1 | 0.998546 | 0.696376 | 0.624852 | 0.749133 | 0.848531 | 0.988636 | 0.375000 | 0.223693 | no |
