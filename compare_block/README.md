# Churn Analysis (Compare)
This is used to compare multiple algorithms on the basis of some conditions to find out the champion model

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