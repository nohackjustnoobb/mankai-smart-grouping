# Training and evaluation report

## Run summary

| Field                             | Value                            |
| --------------------------------- | -------------------------------- |
| Run                               | run_20260824T072405_051871Z      |
| Status                            | Completed                        |
| Started                           | 2026-08-24T07:24:05.051871+00:00 |
| Completed                         | 2026-08-24T08:00:15.684257+00:00 |
| Device                            | cuda                             |
| Model                             | mobilenetv3_large_100            |
| Epochs                            | 10 / 30                          |
| Best epoch                        | 3                                |
| Best validation deployment PR-AUC | 0.8890                           |
| Training time                     | 33m 11s                          |

## Training configuration

| Setting                      | Value    |
| ---------------------------- | -------- |
| Batch size                   | 64       |
| Learning rate                | 3.00e-04 |
| Weight decay                 | 1.00e-04 |
| Embedding dimension          | 256      |
| Classifier hidden dimension  | 256      |
| Samples per epoch            | 70,760   |
| Training positive fraction   | 0.5000   |
| Deployment positive fraction | 0.0100   |
| Random seed                  | 42       |

## Dataset

| Split       | Samples | Positive | Negative |
| ----------- | ------- | -------- | -------- |
| Training    | 601,460 | 35,380   | 566,080  |
| Validation  | 85,918  | 5,054    | 80,864   |
| Calibration | 85,918  | 5,054    | 80,864   |
| Test        | 85,918  | 5,054    | 80,864   |

## Training history

| Epoch | Phase           | Loss    | Learning rate | Validation ROC-AUC | Validation AP | Validation PR-AUC@deploy | Time   | New best |
| ----- | --------------- | ------- | ------------- | ------------------ | ------------- | ------------------------ | ------ | -------- |
| 1     | backbone frozen | 0.22521 | 3.00e-04      | 0.9822             | 0.8504        | 0.6331                   | 2m 57s | ✓        |
| 2     | backbone frozen | 0.14675 | 2.99e-04      | 0.9859             | 0.8757        | 0.6833                   | 2m 55s | ✓        |
| 3     | fine tuning     | 0.09638 | 2.97e-04      | 0.9975             | 0.9695        | 0.8890                   | 3m 26s | ✓        |
| 4     | fine tuning     | 0.02703 | 2.93e-04      | 0.9947             | 0.9409        | 0.8074                   | 3m 24s |          |
| 5     | fine tuning     | 0.01190 | 2.87e-04      | 0.9845             | 0.8753        | 0.6982                   | 3m 25s |          |
| 6     | fine tuning     | 0.00926 | 2.80e-04      | 0.9829             | 0.8557        | 0.6477                   | 3m 24s |          |
| 7     | fine tuning     | 0.00953 | 2.71e-04      | 0.9797             | 0.8453        | 0.6320                   | 3m 25s |          |
| 8     | fine tuning     | 0.00563 | 2.61e-04      | 0.9749             | 0.7769        | 0.4887                   | 3m 25s |          |
| 9     | fine tuning     | 0.00654 | 2.50e-04      | 0.9842             | 0.8635        | 0.6493                   | 3m 25s |          |
| 10    | fine tuning     | 0.00403 | 2.38e-04      | 0.9645             | 0.7226        | 0.4196                   | 3m 25s |          |

## Evaluation

| Split       | Samples | ROC-AUC | AP     | PR-AUC@deploy | Precision@deploy | Recall | F1@deploy | FPR      | Threshold |
| ----------- | ------- | ------- | ------ | ------------- | ---------------- | ------ | --------- | -------- | --------- |
| Calibration | 85,918  | 0.9978  | 0.9715 | 0.8910        | 0.8103           | 0.8419 | 0.8258    | 0.001991 | 0.351780  |
| Test        | 85,918  | 0.9977  | 0.9730 | 0.9024        | 0.8106           | 0.8385 | 0.8244    | 0.001979 | 0.351780  |

### Confusion matrix counts

| Split       | True positive | False positive | True negative | False negative |
| ----------- | ------------- | -------------- | ------------- | -------------- |
| Calibration | 4,255         | 161            | 80,703        | 799            |
| Test        | 4,238         | 160            | 80,704        | 816            |

Threshold source: **calibration**.
