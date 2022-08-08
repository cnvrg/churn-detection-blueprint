# Data Preprocessing
This library performs data preprocessing on tabular data and makes it ready to be used by the classification algorithms. Roughly, the steps include missing value treatment, encoding, evaluation of the structure//type of data and scaling. Below are the relevant features of the library

### Features
- print a statistical summary of the data
- perform missing value treatment for each type of column in the data	
- define the mapping function to map yes and no's to 1's and 0's wherever required	
- missing value treatment based on certain rules

| Blank/Single Valued Columns :- Removed |
| -------------------------------------- |
| Numerical Columns + Unique Value Count > 20% :- Mean |
| Numerical Columns + Unique Value Count < 20% :- Median |
| Numerica Columns + 1/0 Values :- Randomly filled with 1 or 0 |
| String Columns + Yes/No Values :- Randomly filled with Yes or No |
| Other String Columns :- Randomly filled with value "Garbage Value 999" |

- iter over the float64 and int64 columns and check the sparsity of the resultant columns to check which algorithm would be the best fit for the data	
- create charts for each column [X vs y charts, separate for both string and numerical cols]	
- see if the user has defined minmax or standard scaler to be used	
- Perform encoding of character values 	

| If column type ~ string and not in label_encoding_cols :- One Hot Encoding |
| -------------------------------------- |
| If column type ~ string and column values are "Yes" and "No" then :- 1,0 Mapping |
| Other String columns :- Label encoding |

- mark missing values of the dataframe as grabage values and list the indexes where these values lie 
- If id column and label encoding columns are labelled as "None" then they need to be done in all libraries where they are called from
- for one hot encoding columns/ ordinal encoding columns perform the encoding on the train dataset and dump the encoders in the atrifacts to be reused for the user's dataset 
- In case, the user does not want to give label_encoding columns or there is no id_column that exists in the dataset, the user needs to fill those parameters as 'None'
- recommend the best fit algo for the usecase on the basis of sparsity, dimenstionality and categorical percentage	

| Observation | Recommendation | Comment |
| -------------------------------------- | ---------------- | ---- |
| sparsity > 0.5 and percentage_cat_var < 0.25 | Logistic Regression | Sparsity is high and categorical features are low |
| percentage_cat_var < 0.15 | Random Forest | Less categorical variables |
| dimensionality_ratio < 1: | Support Vector Machine | Either decrease features or increase observations |
| percentage_cat_var > 0.5 | Naive Bayes | High number of categorical variables |
| no observatrion | Any | No Comment |

- print the processed data file, a dataframe containing the list of id & label encoding columns/recommendation file/standard scaler object	

# Input Arguments
- `--churn_data` (/data/churn_data/churn.csv) -- raw data uploaded by the user on the platform
- `--id_column`(customerID) -- get the name of the id column by the user
- `--label_encoding_cols`(PaymentMethod,InternetService) -- get list of columns to be label encoded by the user
- `--scaler`(Minmax) -- set the type of scaler to be used
- `--do_scaling` -- whether you want the scaling to be done or not

# Model Artifacts
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
python3 data_preprocessing/data_preprocessing.py --churn_data /data/churn_data/churn.csv --label_encoding "PaymentMethod,InternetService" --scaler Minmax --id_column "customerID" --do_scaling "yes"
```

## About Data Preprocessing
Data Preprocessing librarty serves as a preprocessing tool that has to be run before using the algorithms for the churn analysis to make sure that the input gets converted into a standard format.
a. missing value treatment (https://en.wikipedia.org/wiki/Missing_data)
b. ordinal encoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
c. one hot encoder (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
