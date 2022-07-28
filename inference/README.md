# Churn Analysis (Endpoint)
Churn analysis is the process of using data to understand why your customers have stopped using your product or service.
Analyzing your churn doesn’t only mean knowing what your churn rate is. It’s about figuring out why customers are churning at the rate they are, and how to fix the problem.
It’s one thing to know that you have a 13% churn rate. But unless you understand which customers are cancelling, why their cancelling, when they’re cancelling, and other data points, it’s really hard to improve, that’s why churn analysis is so important for subscription businesses

## Input Arguments

## Features
- input is given as json files of 1 individual (1 row of independent variables/features)    
- input is read in line by line
- missing values are filled by fixed values from output artifacts of train rather than ad-hoc
- encoders and scalers are used from the output artifacts of training blueprint
- final values are tranformed inot a dictionary where the output prediction probability and its round off versiona fter applying on threshold are given respective encoders from the artifacts 
- threshold is fixed at 0.5

## Model Artifacts
- `--data_new` new dataset with encoded and clean values
- `--final_result` final dataset with predictions

## How to run
```
python3 batch_predict/batch_predict.py --datafile /data/churn_data/data_new.csv --model_dir /data/churn_data/my_model.sav --oh_encoder /data/churn_data/encoded_values_file --label_encoder_file /data/churn_data/ordinal_enc --scaler  /data/churn_data/std_scaler.bin --columns_list /data/churn_data/columns_list.csv --threshold 0.6 --processed_file_col /data/churn_data/processed_col.csv
```

## References
Inference library serves as a preprocessing tool and a model loading library that has to be run after running the churn prediction training algorithms for the churn analysis to make sure that the input gets converted into a standard format.
a. missing value treatment (https://en.wikipedia.org/wiki/Missing_data)
b. ordinal encoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
c. one hot encoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
d. dump/load library (https://pypi.org/project/joblib/)
