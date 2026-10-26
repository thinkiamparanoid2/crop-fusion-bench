# E3 Corruption Extension Summary

Generated from saved metrics after the supervisor-requested E3 corrupted-data evaluation. No E3 training or fine-tuning was run.

## Output Coverage

```text
all_metrics.csv rows: 361
E3 metrics rows: 124
robustness_summary.csv rows: 210
```

## Clean Sanity Metrics

| model            | condition   |   accuracy |   macro_f1 |   macro_recall |   early_stage_recall |      ece | clean_reference_condition   |
|:-----------------|:------------|-----------:|-----------:|---------------:|---------------------:|---------:|:----------------------------|
| e2_8             | clean       |   0.987515 |   0.973356 |       0.971295 |             0.981161 | 0.248338 | clean                       |
| e2_8             | clean_regen |   0.987689 |   0.974019 |       0.971457 |             0.982668 | 0.248544 | clean_regen                 |
| e3_8_attack_best | clean       |   0.985261 |   0.968655 |       0.968186 |             0.975132 | 0.108665 | clean                       |
| e3_8_attack_best | clean_regen |   0.985781 |   0.969652 |       0.968968 |             0.976639 | 0.109107 | clean                       |
| e3_8_vqc         | clean       |   0.973643 |   0.943716 |       0.944777 |             0.987189 | 0.118487 | clean                       |
| e3_8_vqc         | clean_regen |   0.973643 |   0.943423 |       0.945127 |             0.988696 | 0.118592 | clean                       |

## Mean Robustness Drops

Macro-F1 drop is computed from each model's clean reference. For regenerated-embedding conditions, E2/E3 use clean_regen as reference.

| model            | condition            | corruption_type     |   n_runs |   macro_f1_mean |   macro_f1_drop_mean |   accuracy_mean |   accuracy_drop_mean |
|:-----------------|:---------------------|:--------------------|---------:|----------------:|---------------------:|----------------:|---------------------:|
| e2_8             | both_corrupted       | defocus_blur        |        1 |      0.16283    |          0.811189    |       0.174614  |          0.813074    |
| e2_8             | both_corrupted       | defocus_blur        |        1 |      0.00590708 |          0.968112    |       0.0293047 |          0.958384    |
| e2_8             | both_corrupted       | defocus_blur        |        1 |      0.00872536 |          0.965293    |       0.0287845 |          0.958904    |
| e3_8_attack_best | both_corrupted       | defocus_blur        |        1 |      0.0561177  |          0.912538    |       0.097451  |          0.88781     |
| e3_8_attack_best | both_corrupted       | defocus_blur        |        1 |      0.0122925  |          0.956363    |       0.0244495 |          0.960812    |
| e3_8_attack_best | both_corrupted       | defocus_blur        |        1 |      0.0258719  |          0.942783    |       0.0346801 |          0.950581    |
| e3_8_vqc         | both_corrupted       | defocus_blur        |        1 |      0.09386    |          0.849856    |       0.101959  |          0.871684    |
| e3_8_vqc         | both_corrupted       | defocus_blur        |        1 |      0.0125082  |          0.931208    |       0.0183804 |          0.955263    |
| e3_8_vqc         | both_corrupted       | defocus_blur        |        1 |      0.0247068  |          0.919009    |       0.0456043 |          0.928039    |
| e2_8             | both_corrupted       | gaussian_noise      |        3 |      0.538199   |          0.43582     |       0.661176  |          0.326513    |
| e2_8             | both_corrupted       | gaussian_noise      |        3 |      0.280504   |          0.693514    |       0.426045  |          0.561644    |
| e2_8             | both_corrupted       | gaussian_noise      |        3 |      0.0617747  |          0.912244    |       0.0956592 |          0.892029    |
| e3_8_attack_best | both_corrupted       | gaussian_noise      |        3 |      0.365779   |          0.602876    |       0.59407   |          0.391191    |
| e3_8_attack_best | both_corrupted       | gaussian_noise      |        3 |      0.168445   |          0.80021     |       0.348997  |          0.636264    |
| e3_8_attack_best | both_corrupted       | gaussian_noise      |        3 |      0.0531784  |          0.915477    |       0.103636  |          0.881625    |
| e3_8_vqc         | both_corrupted       | gaussian_noise      |        3 |      0.383942   |          0.559774    |       0.597885  |          0.375759    |
| e3_8_vqc         | both_corrupted       | gaussian_noise      |        3 |      0.167652   |          0.776064    |       0.293567  |          0.680076    |
| e3_8_vqc         | both_corrupted       | gaussian_noise      |        3 |      0.0429121  |          0.900804    |       0.0593607 |          0.914282    |
| e2_8             | image_corrupted_only | brightness_decrease |        1 |      0.973005   |          0.00101366  |       0.986995  |          0.000693602 |
| e2_8             | image_corrupted_only | brightness_decrease |        1 |      0.969048   |          0.00497115  |       0.985434  |          0.0022542   |
| e2_8             | image_corrupted_only | brightness_decrease |        1 |      0.930969   |          0.0430493   |       0.969828  |          0.0178602   |
| e3_8_attack_best | image_corrupted_only | brightness_decrease |        1 |      0.967059   |          0.00159654  |       0.984567  |          0.000693602 |
| e3_8_attack_best | image_corrupted_only | brightness_decrease |        1 |      0.958977   |          0.00967852  |       0.98162   |          0.00364141  |
| e3_8_attack_best | image_corrupted_only | brightness_decrease |        1 |      0.902364   |          0.0662915   |       0.955089  |          0.0301717   |
| e3_8_vqc         | image_corrupted_only | brightness_decrease |        1 |      0.936699   |          0.00701764  |       0.971389  |          0.0022542   |
| e3_8_vqc         | image_corrupted_only | brightness_decrease |        1 |      0.928506   |          0.0152106   |       0.970522  |          0.00312121  |
| e3_8_vqc         | image_corrupted_only | brightness_decrease |        1 |      0.876943   |          0.0667729   |       0.955436  |          0.018207    |
| e2_8             | image_corrupted_only | brightness_increase |        1 |      0.974139   |         -0.000119805 |       0.987862  |         -0.0001734   |
| e2_8             | image_corrupted_only | brightness_increase |        1 |      0.969711   |          0.00430793  |       0.985088  |          0.00260101  |
| e2_8             | image_corrupted_only | brightness_increase |        1 |      0.96166    |          0.0123586   |       0.981099  |          0.00658921  |
| e3_8_attack_best | image_corrupted_only | brightness_increase |        1 |      0.96943    |         -0.00077433  |       0.984914  |          0.000346801 |
| e3_8_attack_best | image_corrupted_only | brightness_increase |        1 |      0.959936   |          0.0087189   |       0.980579  |          0.00468181  |
| e3_8_attack_best | image_corrupted_only | brightness_increase |        1 |      0.945115   |          0.0235403   |       0.97399   |          0.011271    |
| e3_8_vqc         | image_corrupted_only | brightness_increase |        1 |      0.93905    |          0.00466677  |       0.971562  |          0.0020808   |
| e3_8_vqc         | image_corrupted_only | brightness_increase |        1 |      0.936844   |          0.00687207  |       0.970175  |          0.00346801  |
| e3_8_vqc         | image_corrupted_only | brightness_increase |        1 |      0.926774   |          0.0169427   |       0.965493  |          0.00814982  |
| e2_8             | image_corrupted_only | contrast            |        1 |      0.892137   |          0.0818821   |       0.954049  |          0.0336397   |
| e2_8             | image_corrupted_only | contrast            |        1 |      0.863686   |          0.110332    |       0.93983   |          0.0478585   |
| e2_8             | image_corrupted_only | contrast            |        1 |      0.605415   |          0.368604    |       0.813248  |          0.174441    |
| e3_8_attack_best | image_corrupted_only | contrast            |        1 |      0.813269   |          0.155386    |       0.923184  |          0.0620773   |
| e3_8_attack_best | image_corrupted_only | contrast            |        1 |      0.760214   |          0.208441    |       0.899948  |          0.085313    |
| e3_8_attack_best | image_corrupted_only | contrast            |        1 |      0.438179   |          0.530476    |       0.745622  |          0.239639    |
| e3_8_vqc         | image_corrupted_only | contrast            |        1 |      0.823755   |          0.119961    |       0.934801  |          0.0388417   |
| e3_8_vqc         | image_corrupted_only | contrast            |        1 |      0.781981   |          0.161736    |       0.919022  |          0.0546211   |
| e3_8_vqc         | image_corrupted_only | contrast            |        1 |      0.502169   |          0.441548    |       0.800243  |          0.1734      |
| e2_8             | image_corrupted_only | defocus_blur        |        1 |      0.672674   |          0.301345    |       0.824345  |          0.163343    |
| e2_8             | image_corrupted_only | defocus_blur        |        1 |      0.542824   |          0.431194    |       0.734524  |          0.253165    |
| e2_8             | image_corrupted_only | defocus_blur        |        1 |      0.485556   |          0.488463    |       0.656147  |          0.331542    |
| e3_8_attack_best | image_corrupted_only | defocus_blur        |        1 |      0.436886   |          0.531769    |       0.71493   |          0.270331    |
| e3_8_attack_best | image_corrupted_only | defocus_blur        |        1 |      0.281067   |          0.687589    |       0.565979  |          0.419282    |
| e3_8_attack_best | image_corrupted_only | defocus_blur        |        1 |      0.218693   |          0.749962    |       0.551413  |          0.433848    |
| e3_8_vqc         | image_corrupted_only | defocus_blur        |        1 |      0.571764   |          0.371953    |       0.832842  |          0.140801    |
| e3_8_vqc         | image_corrupted_only | defocus_blur        |        1 |      0.456435   |          0.487281    |       0.765563  |          0.20808     |
| e3_8_vqc         | image_corrupted_only | defocus_blur        |        1 |      0.292522   |          0.651194    |       0.604474  |          0.369169    |
| e2_8             | image_corrupted_only | gaussian_noise      |        3 |      0.647283   |          0.326736    |       0.762268  |          0.22542     |
| e2_8             | image_corrupted_only | gaussian_noise      |        3 |      0.479816   |          0.494203    |       0.597075  |          0.390613    |
| e2_8             | image_corrupted_only | gaussian_noise      |        3 |      0.324054   |          0.649965    |       0.488931  |          0.498757    |
| e3_8_attack_best | image_corrupted_only | gaussian_noise      |        3 |      0.455991   |          0.512664    |       0.730016  |          0.255245    |
| e3_8_attack_best | image_corrupted_only | gaussian_noise      |        3 |      0.193882   |          0.774773    |       0.501532  |          0.483729    |
| e3_8_attack_best | image_corrupted_only | gaussian_noise      |        3 |      0.125955   |          0.8427      |       0.37992   |          0.605341    |
| e3_8_vqc         | image_corrupted_only | gaussian_noise      |        3 |      0.544252   |          0.399464    |       0.795099  |          0.178545    |
| e3_8_vqc         | image_corrupted_only | gaussian_noise      |        3 |      0.272543   |          0.671173    |       0.513728  |          0.459916    |
| e3_8_vqc         | image_corrupted_only | gaussian_noise      |        3 |      0.216875   |          0.726841    |       0.437663  |          0.535981    |
| e2_8             | image_corrupted_only | jpeg_compression    |        1 |      0.862917   |          0.111102    |       0.939136  |          0.0485521   |
| e2_8             | image_corrupted_only | jpeg_compression    |        1 |      0.842219   |          0.131799    |       0.927692  |          0.0599965   |
| e2_8             | image_corrupted_only | jpeg_compression    |        1 |      0.717348   |          0.256671    |       0.854864  |          0.132825    |
| e3_8_attack_best | image_corrupted_only | jpeg_compression    |        1 |      0.774678   |          0.193977    |       0.906884  |          0.078377    |
| e3_8_attack_best | image_corrupted_only | jpeg_compression    |        1 |      0.736949   |          0.231706    |       0.887463  |          0.0977978   |
| e3_8_attack_best | image_corrupted_only | jpeg_compression    |        1 |      0.551716   |          0.416939    |       0.773886  |          0.211375    |
| e3_8_vqc         | image_corrupted_only | jpeg_compression    |        1 |      0.779945   |          0.163771    |       0.911566  |          0.0620773   |
| e3_8_vqc         | image_corrupted_only | jpeg_compression    |        1 |      0.748795   |          0.194921    |       0.902722  |          0.0709208   |
| e3_8_vqc         | image_corrupted_only | jpeg_compression    |        1 |      0.634067   |          0.309649    |       0.857291  |          0.116352    |
| e2_8             | image_corrupted_only | motion_blur         |        1 |      0.774132   |          0.199886    |       0.889371  |          0.098318    |
| e2_8             | image_corrupted_only | motion_blur         |        1 |      0.774132   |          0.199886    |       0.889371  |          0.098318    |
| e2_8             | image_corrupted_only | motion_blur         |        1 |      0.727187   |          0.246832    |       0.862667  |          0.125022    |
| e3_8_attack_best | image_corrupted_only | motion_blur         |        1 |      0.672621   |          0.296034    |       0.794521  |          0.19074     |
| e3_8_attack_best | image_corrupted_only | motion_blur         |        1 |      0.672621   |          0.296034    |       0.794521  |          0.19074     |
| e3_8_attack_best | image_corrupted_only | motion_blur         |        1 |      0.584134   |          0.384521    |       0.74094   |          0.244321    |
| e3_8_vqc         | image_corrupted_only | motion_blur         |        1 |      0.671983   |          0.271733    |       0.871164  |          0.10248     |
| e3_8_vqc         | image_corrupted_only | motion_blur         |        1 |      0.671983   |          0.271733    |       0.871164  |          0.10248     |
| e3_8_vqc         | image_corrupted_only | motion_blur         |        1 |      0.584448   |          0.359268    |       0.843593  |          0.13005     |
| e2_8             | iot_corrupted_only   | drift               |        1 |      0.716364   |          0.257654    |       0.553494  |          0.434195    |
| e2_8             | iot_corrupted_only   | drift               |        1 |      0.396485   |          0.577534    |       0.182591  |          0.805098    |
| e2_8             | iot_corrupted_only   | drift               |        1 |      0.0603263  |          0.913692    |       0.0419629 |          0.945726    |
| e3_8_attack_best | iot_corrupted_only   | drift               |        1 |      0.623389   |          0.345267    |       0.581585  |          0.403676    |
| e3_8_attack_best | iot_corrupted_only   | drift               |        1 |      0.285051   |          0.683604    |       0.20652   |          0.778741    |
| e3_8_attack_best | iot_corrupted_only   | drift               |        1 |      0.0682466  |          0.900409    |       0.0707474 |          0.914514    |
| e3_8_vqc         | iot_corrupted_only   | drift               |        1 |      0.390989   |          0.552727    |       0.432807  |          0.540836    |
| e3_8_vqc         | iot_corrupted_only   | drift               |        1 |      0.137377   |          0.806339    |       0.0860066 |          0.887637    |
| e3_8_vqc         | iot_corrupted_only   | drift               |        1 |      0.0538011  |          0.889915    |       0.0362407 |          0.937402    |
| e2_8             | iot_corrupted_only   | dropout             |        3 |      0.875616   |          0.0984027   |       0.926016  |          0.0616727   |
| e2_8             | iot_corrupted_only   | dropout             |        3 |      0.805246   |          0.168772    |       0.871915  |          0.115774    |
| e2_8             | iot_corrupted_only   | dropout             |        3 |      0.715728   |          0.258291    |       0.801341  |          0.186348    |
| e3_8_attack_best | iot_corrupted_only   | dropout             |        3 |      0.80646    |          0.162195    |       0.895324  |          0.089937    |
| e3_8_attack_best | iot_corrupted_only   | dropout             |        3 |      0.71126    |          0.257396    |       0.827813  |          0.157448    |
| e3_8_attack_best | iot_corrupted_only   | dropout             |        3 |      0.614904   |          0.353751    |       0.74198   |          0.243281    |
| e3_8_vqc         | iot_corrupted_only   | dropout             |        3 |      0.723553   |          0.220164    |       0.865095  |          0.108549    |
| e3_8_vqc         | iot_corrupted_only   | dropout             |        3 |      0.580227   |          0.363489    |       0.77036   |          0.203283    |
| e3_8_vqc         | iot_corrupted_only   | dropout             |        3 |      0.428752   |          0.514964    |       0.639212  |          0.334432    |
| e2_8             | iot_corrupted_only   | sensor_noise        |        3 |      0.958826   |          0.015193    |       0.974973  |          0.012716    |
| e2_8             | iot_corrupted_only   | sensor_noise        |        3 |      0.903177   |          0.070842    |       0.927287  |          0.0604011   |
| e2_8             | iot_corrupted_only   | sensor_noise        |        3 |      0.427597   |          0.546421    |       0.397607  |          0.590081    |
| e3_8_attack_best | iot_corrupted_only   | sensor_noise        |        3 |      0.928039   |          0.0406158   |       0.960696  |          0.0245651   |
| e3_8_attack_best | iot_corrupted_only   | sensor_noise        |        3 |      0.838084   |          0.130571    |       0.899139  |          0.0861222   |
| e3_8_attack_best | iot_corrupted_only   | sensor_noise        |        3 |      0.350705   |          0.61795     |       0.361424  |          0.623837    |
| e3_8_vqc         | iot_corrupted_only   | sensor_noise        |        3 |      0.868695   |          0.0750215   |       0.936304  |          0.0373389   |
| e3_8_vqc         | iot_corrupted_only   | sensor_noise        |        3 |      0.720145   |          0.223571    |       0.834287  |          0.139356    |
| e3_8_vqc         | iot_corrupted_only   | sensor_noise        |        3 |      0.206768   |          0.736948    |       0.234669  |          0.738975    |

## Extra Macro-F1 Drop Versus E2-8

Positive values mean the E3 model degraded more than E2-8 for that condition/corruption type.

| condition            | corruption_type     |       e2_8 |   e3_8_attack_best |   e3_8_vqc |   e3_8_vqc_extra_drop_vs_e2_8 |   e3_8_attack_best_extra_drop_vs_e2_8 |
|:---------------------|:--------------------|-----------:|-------------------:|-----------:|------------------------------:|--------------------------------------:|
| both_corrupted       | defocus_blur        | 0.914865   |          0.937228  | 0.900025   |                   -0.0148399  |                            0.0223633  |
| both_corrupted       | gaussian_noise      | 0.680526   |          0.772854  | 0.745548   |                    0.0650215  |                            0.0923283  |
| image_corrupted_only | brightness_decrease | 0.0163447  |          0.0258555 | 0.029667   |                    0.0133223  |                            0.0095108  |
| image_corrupted_only | brightness_increase | 0.00551558 |          0.010495  | 0.00949384 |                    0.00397826 |                            0.00497938 |
| image_corrupted_only | contrast            | 0.186939   |          0.298101  | 0.241081   |                    0.0541421  |                            0.111162   |
| image_corrupted_only | defocus_blur        | 0.407001   |          0.65644   | 0.503476   |                    0.0964754  |                            0.249439   |
| image_corrupted_only | gaussian_noise      | 0.490301   |          0.710046  | 0.599159   |                    0.108858   |                            0.219744   |
| image_corrupted_only | jpeg_compression    | 0.166524   |          0.280874  | 0.222781   |                    0.0562565  |                            0.11435    |
| image_corrupted_only | motion_blur         | 0.215535   |          0.32553   | 0.300912   |                    0.0853766  |                            0.109995   |
| iot_corrupted_only   | drift               | 0.58296    |          0.643093  | 0.74966    |                    0.1667     |                            0.0601329  |
| iot_corrupted_only   | dropout             | 0.175155   |          0.257781  | 0.366206   |                    0.19105    |                            0.0826255  |
| iot_corrupted_only   | sensor_noise        | 0.210819   |          0.263046  | 0.34518    |                    0.134361   |                            0.0522269  |

## Attack E3 Drop Minus Original E3 Drop

Negative values mean `e3_8_attack_best` degraded less than original `e3_8_vqc`; positive values mean it degraded more.

| condition            | corruption_type     |   attack_minus_original_drop |
|:---------------------|:--------------------|-----------------------------:|
| both_corrupted       | defocus_blur        |                   0.0372032  |
| both_corrupted       | gaussian_noise      |                   0.0273068  |
| image_corrupted_only | brightness_decrease |                  -0.00381154 |
| image_corrupted_only | brightness_increase |                   0.00100112 |
| image_corrupted_only | contrast            |                   0.0570196  |
| image_corrupted_only | defocus_blur        |                   0.152964   |
| image_corrupted_only | gaussian_noise      |                   0.110886   |
| image_corrupted_only | jpeg_compression    |                   0.0580935  |
| image_corrupted_only | motion_blur         |                   0.0246183  |
| iot_corrupted_only   | drift               |                  -0.106567   |
| iot_corrupted_only   | dropout             |                  -0.108425   |
| iot_corrupted_only   | sensor_noise        |                  -0.0821345  |

## Interpretation

- E3 was evaluated on corrupted TEST embeddings only; no training was run.
- The clean E3 attack checkpoint is stronger than original E3-8, but robustness must be judged by Macro-F1 drop, not clean Macro-F1 alone.
- Compare E3 against `e2_8` first because `e2_8` is the matched classical 8D comparator used in the corruption harness.
- A quantum robustness improvement should only be claimed if an E3 model has smaller Macro-F1 drops than `e2_8` under the same corrupted-input protocol.
