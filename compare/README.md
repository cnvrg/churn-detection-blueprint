# Churn Analysis (Compare)
This library is used to compare multiple algorithms on the basis of accuracy score, you can change it based on your requirements, to pick the champion model
Supported Metrics:
1. `accuracy_score`
2. `precision`
3. `recall_score`
4. `f1_score`

## Features
- get comparison parameters (overfit_magnitude, prec_rec_metric, avg_train_metric, avg_test_metric) form the evaluation file saved in the artifacts 
- comapre the data values on the basis of these parameters and suggest an algorithm
- save the evaluation metrics with recommendations in the atrifacts

## Model Artifacts
- `--eval_1` recommendation evalutaion metrics
- `--churn_output.csv` churn predictions from the best model

## How to run
```
python3 compare_block/compare.py
```