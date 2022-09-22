# Churn Detection (Batch-Predict)
Churn analysis is the process of using data to understand why your customers have stopped using your product or service.
Analyzing your churn doesn’t only mean knowing what your churn rate is. It’s about figuring out why customers are churning at the rate they are, and how to fix the problem.
It’s one thing to know that you have a 13% churn rate. But unless you understand which customers are cancelling, why their cancelling, when they’re cancelling, and other data points, it’s really hard to improve, that’s why churn analysis is so important for subscription businesses.
This library will just predict the churn probability from pretrained and saved models and artifacts.

## Input Arguments

## Features
- input is given as as csv file of user characterstics (independent variables like demographics, transaction etc) but only specific to the data that has been used in training.
- missing values are filled by fixed values from output artifacts of train rather than ad-hoc
- encoders and scalers are used from the output artifacts of training blueprint (that have been downloaded and pushed to s3)
- final values are stored in a dictionary
- threshold values can be given by user

## Model Artifacts
- `--recommendation file` file which contains recommendations on what algorithm to use

| Sparsity	| Categorical_Percentage	| Dimensionality |	Recommendation	| General Comment |
|---|---|---|---|---|
| 0.314780633252875	| 0.428571428571429	| 185.342105263158	| Any	| No comment |

- `--original_col.csv`	original file column names and average/random values as a single row

| customerID  | gender  | SeniorCitizen  | Partner  | Dependents  | tenure  | PhoneService  | MultipleLines  | InternetService  | OnlineSecurity  | OnlineBackup        | DeviceProtection  | TechSupport  | StreamingTV  | StreamingMovies  | Contract       | PaperlessBilling  | PaymentMethod           | MonthlyCharges    | TotalCharges       | Churn  |
|-------------|---------|----------------|----------|-------------|---------|---------------|----------------|------------------|-----------------|---------------------|-------------------|--------------|--------------|------------------|----------------|-------------------|-------------------------|-------------------|--------------------|--------|
| 1867-TJHTS  | Female  | 0              | Yes      | Yes         | 29      | Yes           | No             | Fiber optic      | No              | No internet service | No                | No           | No           | Yes              | Month-to-month | Yes               | Credit card (automatic) | 64.76169246059918 | 2283.3004408418656 | Yes    |

- `--data_df.csv`	processed file (after one hot encoding, label encoding, missing value treatment)

| customerID  | SeniorCitizen  | Partner  | Dependents  | tenure  | PhoneService  | InternetService  | PaperlessBilling  | PaymentMethod  | MonthlyCharges      | TotalCharges         | Churn  | Contract-Month-to-month  | Contract-One year  | Contract-Two year  | DeviceProtection-No  | DeviceProtection-No internet service  | DeviceProtection-Yes  | MultipleLines-No  | MultipleLines-No phone service  | MultipleLines-Yes  | OnlineBackup-No  | OnlineBackup-No internet service  | OnlineBackup-Yes  | OnlineSecurity-No  | OnlineSecurity-No internet service  | OnlineSecurity-Yes  | StreamingMovies-No  | StreamingMovies-No internet service  | StreamingMovies-Yes  | StreamingTV-No  | StreamingTV-No internet service  | StreamingTV-Yes  | TechSupport-No  | TechSupport-No internet service  | TechSupport-Yes  | gender-Female  | gender-Male  |
|-------------|----------------|----------|-------------|---------|---------------|------------------|-------------------|----------------|---------------------|----------------------|--------|--------------------------|--------------------|--------------------|----------------------|---------------------------------------|-----------------------|-------------------|---------------------------------|--------------------|------------------|-----------------------------------|-------------------|--------------------|-------------------------------------|---------------------|---------------------|--------------------------------------|----------------------|-----------------|----------------------------------|------------------|-----------------|----------------------------------|------------------|----------------|--------------|
| 1867-TJHTS  | 0.0            | 1        | 1           | 1.0     | 1             | 1.0              | 1                 | 1.0            | 0.11542288557213931 | 0.001275098084468036 | 1      | 1.0                      | 0.0                | 0.0                | 1.0                  | 0.0                                   | 0.0                   | 1.0               | 0.0                             | 0.0                | 0.0              | 1.0                               | 0.0               | 1.0                | 0.0                                 | 0.0                 | 0.0                 | 0.0                                  | 1.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 1.0            | 0.0          |
| 5575-GNVDE  | 0.0            | 0        | 0           | 34.0    | 1             | 0.0              | 0                 | 3.0            | 0.3850746268656716  | 0.21586660512347106  | 0      | 0.0                      | 1.0                | 0.0                | 0.0                  | 0.0                                   | 1.0                   | 1.0               | 0.0                             | 0.0                | 1.0              | 0.0                               | 0.0               | 0.0                | 0.0                                 | 1.0                 | 1.0                 | 0.0                                  | 0.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 0.0            | 1.0          |


- `--processed_col.csv`	processed file (after one hot encoding, label encoding, missing value treatment) but with only 1 row (for batch predict)

| customerID  | SeniorCitizen  | Partner  | Dependents  | tenure  | PhoneService  | InternetService  | PaperlessBilling  | PaymentMethod  | MonthlyCharges      | TotalCharges         | Churn  | Contract-Month-to-month  | Contract-One year  | Contract-Two year  | DeviceProtection-No  | DeviceProtection-No internet service  | DeviceProtection-Yes  | MultipleLines-No  | MultipleLines-No phone service  | MultipleLines-Yes  | OnlineBackup-No  | OnlineBackup-No internet service  | OnlineBackup-Yes  | OnlineSecurity-No  | OnlineSecurity-No internet service  | OnlineSecurity-Yes  | StreamingMovies-No  | StreamingMovies-No internet service  | StreamingMovies-Yes  | StreamingTV-No  | StreamingTV-No internet service  | StreamingTV-Yes  | TechSupport-No  | TechSupport-No internet service  | TechSupport-Yes  | gender-Female  | gender-Male  |
|-------------|----------------|----------|-------------|---------|---------------|------------------|-------------------|----------------|---------------------|----------------------|--------|--------------------------|--------------------|--------------------|----------------------|---------------------------------------|-----------------------|-------------------|---------------------------------|--------------------|------------------|-----------------------------------|-------------------|--------------------|-------------------------------------|---------------------|---------------------|--------------------------------------|----------------------|-----------------|----------------------------------|------------------|-----------------|----------------------------------|------------------|----------------|--------------|
| 1867-TJHTS  | 0.0            | 1        | 1           | 1.0     | 1             | 1.0              | 1                 | 1.0            | 0.11542288557213931 | 0.001275098084468036 | 1      | 1.0                      | 0.0                | 0.0                | 1.0                  | 0.0                                   | 0.0                   | 1.0               | 0.0                             | 0.0                | 0.0              | 1.0                               | 0.0               | 1.0                | 0.0                                 | 0.0                 | 0.0                 | 0.0                                  | 1.0                  | 1.0             | 0.0                              | 0.0              | 1.0             | 0.0                              | 0.0              | 1.0            | 0.0          |


- `--recommendation.csv`	file which contains recommendations on what algorithm to use

| Sparsity           | Categorical_Percentage  | Dimensionality    | Recommendation  | General Comment  |
|--------------------|-------------------------|-------------------|-----------------|------------------|
| 0.3147806332528752 | 0.42857142857142855     | 185.3421052631579 | Any             | No comment       | 

- `--ordinal_enc`	label/ordinal encoder saved file after fitting the encoder on training data
- `--one_hot_encoder`	one hot encoder saved file after fitting the encoder on training data
- `--columns_list.csv`	table of 3 columns, one hot encoded columns, label encoded columns and id columns

|    OHE_columns   | id_columns | label_encoded_columns |
|:----------------:|:----------:|:---------------------:|
| Contract         | customerID | InternetService       |
| DeviceProtection |            | PaymentMethod         |

- `--mis_col_type.csv`	columns categorised on what kind of missing value treatment they received, mean, median, random value?

| Mean  | 0-1           | Median  | Yes-No     | String     |
|-------|---------------|---------|------------|------------|
|       | SeniorCitizen |         | Partner    | customerID |
|       |               | tenure  | Dependents | gender     |
- `--std_scaler.bin`	standard scaler saved object after fitting scaler on training data

## How to run
```
import json
request_dict = {'vars':['7590-VHVEG','Female','0','Yes','No','1','No','No phone service','DSL','No','Yes','No','No','No','No','Month-to-month','Yes','Electronic check','29.85','29.85']}
json.dumps(request_dict)

import http.client
conn = http.client.HTTPSConnection("inference-10-1.anx6qra9ysplne3dszkzmbu.staging-cloud.cnvrg.io", 443)
payload = "{\"input_params\":" + json.dumps(request_dict) + "}"
headers = {
    'Cnvrg-Api-Key': "4tPfrGjUY2dEqqqdoz2jNHFH",
    'Content-Type': "application/json"
    }
conn.request("POST", "/api/v1/endpoints/ttyehaaewqckev9wbaky", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
```
## Sample input
```
'7590-VHVEG','Female','0','Yes','No','1','No','No phone service','DSL','No','Yes','No','No','No','No','Month-to-month','Yes','Electronic check','29.85','29.85'
```
These individual values should match the original training input in
1. Number of values. In case the original input has 20 values, you should give 20 values to the inference, comma separated.
2. The order of the values should be the same.
3. The last value (dependent variable) should NOT be in the string you put in the endpoint.
4. For the current example, here is the snapshot of the original training file on which this model was trained on.

|customerID |gender |SeniorCitizen |Partner |Dependents |tenure |PhoneService |MultipleLines |InternetService |OnlineSecurity |OnlineBackup |DeviceProtection |TechSupport |StreamingTV |StreamingMovies |Contract |PaperlessBilling |PaymentMethod |MonthlyCharges |TotalCharges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|7590-VHVEG |Female |0 |Yes |No |1 |No |No phone service |DSL |No |Yes |No |No |No |No |Month-to-month |Yes |Electronic check |29.85 |29.85 |
|4563-VHVEG |Male |1 |Yes |No |1 |No |No phone service |DSL |No |Yes |No |No |Yes |No |Month-to-month |Yes |Electronic check |59.85 |129.85 |

## References
Inference library serves as a preprocessing tool and a model loading library that has to be run after running the churn prediction training algorithms for the churn analysis to make sure that the input gets converted into a standard format.
a. missing value treatment (https://en.wikipedia.org/wiki/Missing_data)
b. ordinal encoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
c. one hot encoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
d. dump/load library (https://pypi.org/project/joblib/)
