# Churn Detection (Endpoint)
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
```
{
  "0": [
    {
      "Churn_Prediction": 0,
      "Churn_Probability": "0.32",
      "customerID": "7590-VHVEG"
    }
  ]
}
```
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
